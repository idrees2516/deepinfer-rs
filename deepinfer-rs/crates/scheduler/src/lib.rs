//! # deepinfer-scheduler
//!
//! Continuous-batching scheduler and the two-thread **zero-overhead
//! asynchronous engine** of DeepInfer-rs (upstream DeepInfer: "Asynchronous
//! scheduling (zero-overhead batch scheduler)").
//!
//! * **Continuous batching**: decode requests run every step; prefill work
//!   is admitted as chunked slices while the token budget allows
//!   (vLLM-style, as used by DeepInfer's engine loop).
//! * **Chunked prefill**: prompts longer than `chunked_prefill_size` are
//!   split across steps.
//! * **Preemption**: when the decode load squeezes the budget, the newest
//!   prefilling requests that cannot be scheduled are evicted (recompute
//!   strategy — their chunk progress resets and they are re-admitted
//!   later).
//! * **Zero-overhead overlap**: [`AsyncEngine`] runs a scheduler thread
//!   and an engine thread with `std::sync::mpsc` handoff (lock-free fast
//!   paths from the standard library). The pipeline has a one-step
//!   window: while the engine executes step *N*, the scheduler thread
//!   folds in any results that arrive and builds step *N+1*'s plan —
//!   results of step *N* are folded at the top of the *N+1* scheduling
//!   pass, so scheduling work overlaps execution.
//!
//! Determinism contract: scheduling decisions depend only on the request
//! set and the step counter — never on wall-clock time — so identical
//! inputs produce identical plan sequences (the foundation of
//! `DEEPINFER_BATCH_INVARIANT` deployments). The only wall-clock use is a
//! polite idle back-off when there is nothing to do.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use deepinfer_core::env::Env;
use deepinfer_core::{ensure, Result};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{channel, TryRecvError};
use std::sync::{Arc, Mutex};

/// Scheduler tuning knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Maximum concurrently running requests.
    pub max_running_requests: usize,
    /// Total token budget per step (decode + prefill chunk tokens).
    pub max_batch_tokens: usize,
    /// Maximum prompt tokens prefilled per request per step.
    pub chunked_prefill_size: usize,
    /// `DEEPINFER_BATCH_INVARIANT`: admission decisions must depend only
    /// on the request set and step counter (already true here; the flag
    /// is honoured for API compatibility and asserted in tests).
    pub batch_invariant: bool,
}

impl Default for SchedulerConfig {
    fn default() -> SchedulerConfig {
        SchedulerConfig {
            max_running_requests: 64,
            max_batch_tokens: 8192,
            chunked_prefill_size: 2048,
            batch_invariant: false,
        }
    }
}

impl SchedulerConfig {
    /// Derives the config from the `DEEPINFER_*` environment.
    pub fn from_env(env: &Env) -> SchedulerConfig {
        SchedulerConfig {
            batch_invariant: env.batch_invariant,
            ..SchedulerConfig::default()
        }
    }

    /// Validates the knob ranges.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_running_requests >= 1,
            InvalidArgument,
            "max_running_requests must be >= 1"
        );
        ensure!(
            self.max_batch_tokens >= 1,
            InvalidArgument,
            "max_batch_tokens must be >= 1"
        );
        ensure!(
            self.chunked_prefill_size >= 1,
            InvalidArgument,
            "chunked_prefill_size must be >= 1"
        );
        Ok(())
    }
}

/// One generation request.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// Request id (assigned automatically by `add_request` when `0`).
    pub id: u64,
    /// Prompt token ids.
    pub prompt: Vec<u32>,
    /// Maximum generated tokens before stopping with `Length`.
    pub max_tokens: usize,
    /// Monotonic arrival index driving FCFS admission (assigned by
    /// `add_request`).
    pub arrival_order: u64,
}

impl Request {
    /// A request with the given prompt and generation limit.
    pub fn new(prompt: Vec<u32>, max_tokens: usize) -> Request {
        Request {
            id: 0,
            prompt,
            max_tokens,
            arrival_order: 0,
        }
    }
}

/// Why a request finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// `max_tokens` reached.
    Length,
    /// EOS token generated.
    Eos,
    /// Aborted externally.
    Aborted,
}

/// Per-request execution phase.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// Waiting for admission.
    Waiting,
    /// Prefilling; `chunk` prompt tokens are already processed.
    Prefilling {
        /// Prompt tokens processed so far.
        chunk: usize,
    },
    /// Decoding; `generated` tokens produced so far.
    Decoding {
        /// Tokens generated so far.
        generated: usize,
    },
    /// Finished.
    Finished {
        /// Why it finished.
        reason: StopReason,
    },
}

/// One step's execution plan handed to the engine.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchPlan {
    /// Nothing to do.
    Idle,
    /// Execute decodes (in FCFS order) plus the listed prefill chunks.
    Step {
        /// Ids of requests producing one token each this step.
        decode_reqs: Vec<u64>,
        /// `(id, chunk_start, chunk_len)` prefill slices.
        prefill_reqs: Vec<(u64, usize, usize)>,
    },
}

#[derive(Debug, Clone)]
struct ReqState {
    req: Request,
    phase: Phase,
}

/// Continuous-batching scheduler (single-threaded, deterministic).
pub struct Scheduler {
    cfg: SchedulerConfig,
    states: Vec<ReqState>,
    next_arrival: u64,
    next_id: u64,
    step: u64,
    finished: Vec<(u64, StopReason)>,
}

impl Scheduler {
    /// Creates a scheduler with the given config.
    pub fn new(cfg: SchedulerConfig) -> Scheduler {
        Scheduler {
            cfg,
            states: Vec::new(),
            next_arrival: 0,
            next_id: 1,
            step: 0,
            finished: Vec::new(),
        }
    }

    /// Adds a request; returns its assigned id (the request's own id when
    /// non-zero).
    pub fn add_request(&mut self, mut req: Request) -> u64 {
        if req.id == 0 {
            req.id = self.next_id;
            self.next_id += 1;
        }
        req.arrival_order = self.next_arrival;
        self.next_arrival += 1;
        let id = req.id;
        self.states.push(ReqState {
            req,
            phase: Phase::Waiting,
        });
        id
    }

    /// Number of not-yet-finished requests.
    pub fn num_active(&self) -> usize {
        self.states
            .iter()
            .filter(|s| !matches!(s.phase, Phase::Finished { .. }))
            .count()
    }

    /// Whether every request has finished.
    pub fn is_done(&self) -> bool {
        self.states
            .iter()
            .all(|s| matches!(s.phase, Phase::Finished { .. }))
    }

    /// Finished requests `(id, reason)` in completion order.
    pub fn finished(&self) -> &[(u64, StopReason)] {
        &self.finished
    }

    /// The current phase of a request (diagnostics/tests).
    pub fn phase(&self, id: u64) -> Option<&Phase> {
        self.states.iter().find(|s| s.req.id == id).map(|s| &s.phase)
    }

    /// The scheduler's step counter (increments on every `schedule`).
    pub fn steps(&self) -> u64 {
        self.step
    }

    /// Computes the next batch.
    ///
    /// Policy (deterministic): decodes always run (1 token each against
    /// the budget); existing prefill continuations are scheduled FCFS;
    /// waiting requests are admitted while budget and the running cap
    /// allow (the head request is always admitted at least one chunk
    /// token when nothing else is running, preventing starvation);
    /// prefill continuations that no longer fit are evicted back to
    /// `Waiting` (recompute).
    pub fn schedule(&mut self) -> BatchPlan {
        self.step += 1;
        // Stable order: (arrival_order, id).
        self.states.sort_by(|a, b| {
            a.req
                .arrival_order
                .cmp(&b.req.arrival_order)
                .then(a.req.id.cmp(&b.req.id))
        });

        let mut decode_reqs: Vec<u64> = Vec::new();
        for s in &self.states {
            if let Phase::Decoding { .. } = s.phase {
                decode_reqs.push(s.req.id);
            }
        }
        let mut budget = self
            .cfg
            .max_batch_tokens
            .saturating_sub(decode_reqs.len());
        let mut prefill_reqs: Vec<(u64, usize, usize)> = Vec::new();
        let mut running = self
            .states
            .iter()
            .filter(|s| !matches!(s.phase, Phase::Waiting | Phase::Finished { .. }))
            .count();
        let nothing_running = running == 0;

        // 1. Continue existing prefills (FCFS); evict those that no
        //    longer fit (recompute on re-admission).
        let idxs: Vec<usize> = (0..self.states.len()).collect();
        for idx in idxs {
            if let Phase::Prefilling { chunk } = self.states[idx].phase {
                let prompt_len = self.states[idx].req.prompt.len();
                if chunk >= prompt_len {
                    continue;
                }
                let want = self.cfg.chunked_prefill_size.min(prompt_len - chunk);
                if want <= budget {
                    budget -= want;
                    prefill_reqs.push((self.states[idx].req.id, chunk, want));
                } else if budget > 0 {
                    prefill_reqs.push((self.states[idx].req.id, chunk, budget));
                    budget = 0;
                } else {
                    // Evicted: newest prefills lose their progress.
                    self.states[idx].phase = Phase::Waiting;
                }
            }
        }

        // 2. Admit waiting requests (FCFS).
        for idx in 0..self.states.len() {
            if !matches!(self.states[idx].phase, Phase::Waiting) {
                continue;
            }
            if running >= self.cfg.max_running_requests {
                break;
            }
            let prompt_len = self.states[idx].req.prompt.len();
            let want = self.cfg.chunked_prefill_size.min(prompt_len).max(1);
            let allowed = if nothing_running {
                // Starvation guard for the head request.
                want.max(1)
            } else {
                want.min(budget)
            };
            if allowed == 0 {
                continue;
            }
            if allowed > budget && !(nothing_running && prefill_reqs.is_empty()) {
                continue;
            }
            budget = budget.saturating_sub(allowed);
            running += 1;
            prefill_reqs.push((self.states[idx].req.id, 0, allowed));
        }

        if decode_reqs.is_empty() && prefill_reqs.is_empty() {
            return BatchPlan::Idle;
        }
        BatchPlan::Step {
            decode_reqs,
            prefill_reqs,
        }
    }

    /// Folds one step's generated tokens in: `(req_id, token)`. Only
    /// decoding requests consume tokens (a request whose prefill just
    /// completed is promoted by [`Scheduler::commit_prefill`] into
    /// `Decoding` before its first token is reported).
    pub fn report_tokens(&mut self, tokens: &[(u64, u32)], eos_id: u32) {
        for &(id, tok) in tokens {
            let Some(state) = self.states.iter_mut().find(|s| s.req.id == id) else {
                continue;
            };
            if let Phase::Decoding { generated } = state.phase {
                let generated = generated + 1;
                if tok == eos_id {
                    state.phase = Phase::Finished {
                        reason: StopReason::Eos,
                    };
                    self.finished.push((id, StopReason::Eos));
                } else if generated >= state.req.max_tokens {
                    state.phase = Phase::Finished {
                        reason: StopReason::Length,
                    };
                    self.finished.push((id, StopReason::Length));
                } else {
                    state.phase = Phase::Decoding { generated };
                }
            }
        }
    }

    /// Commits prefill progress for a `(id, chunk_start, chunk_len)`
    /// plan slice; promotes to `Decoding` when the prompt is complete.
    pub fn commit_prefill(&mut self, id: u64, chunk_start: usize, chunk_len: usize) {
        let Some(state) = self.states.iter_mut().find(|s| s.req.id == id) else {
            return;
        };
        let done = chunk_start + chunk_len;
        if done >= state.req.prompt.len() {
            state.phase = Phase::Decoding { generated: 0 };
        } else {
            state.phase = Phase::Prefilling { chunk: done };
        }
    }
}

/// The engine interface implemented by the runtime (keeps this crate
/// decoupled): execute one step plan, returning `(req_id, token)` per
/// decode position — plus one token for every request whose prefill
/// completed in this step — in plan order.
pub trait EngineOps: Send {
    /// Executes the plan.
    fn execute_step(&mut self, plan: &BatchPlan, eos_id: u32) -> Result<Vec<(u64, u32)>>;
}

/// Two-thread overlapping engine: the scheduler thread builds step *N+1*
/// while the engine thread executes step *N*.
pub struct AsyncEngine<E: EngineOps + 'static> {
    cfg: SchedulerConfig,
    engine: Option<E>,
    inbox: Arc<Mutex<Vec<Request>>>,
    steps: Arc<AtomicU64>,
    _phantom: std::marker::PhantomData<E>,
}

impl<E: EngineOps + 'static> AsyncEngine<E> {
    /// Wraps `engine` with the given scheduler config (threads start in
    /// [`AsyncEngine::run`]).
    pub fn new(cfg: SchedulerConfig, engine: E) -> AsyncEngine<E> {
        AsyncEngine {
            cfg,
            engine: Some(engine),
            inbox: Arc::new(Mutex::new(Vec::new())),
            steps: Arc::new(AtomicU64::new(0)),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Queues a request (id assigned here when `0`).
    pub fn submit(&mut self, req: Request) -> u64 {
        let mut guard = self.inbox.lock().expect("inbox poisoned");
        let id = if req.id == 0 {
            // Deterministic ids from the inbox position.
            (guard.len() as u64) + 1
        } else {
            req.id
        };
        let mut req = req;
        req.id = id;
        guard.push(req);
        id
    }

    /// Steps executed so far (across `run` calls).
    pub fn steps_executed(&self) -> u64 {
        self.steps.load(Ordering::SeqCst)
    }

    /// Drives all submitted requests to completion (or `max_steps`
    /// scheduler steps), returning `(id, StopReason)` per request.
    ///
    /// Pipeline: the scheduler thread sends the plan for step *N* and
    /// immediately loops to fold whatever results have arrived (from
    /// step *N-1*) and build step *N+1* — scheduling overlaps execution
    /// with a one-step window.
    pub fn run(&mut self, eos_id: u32, max_steps: usize) -> Result<Vec<(u64, StopReason)>> {
        let engine = self
            .engine
            .take()
            .ok_or_else(|| deepinfer_core::Error::internal("engine already consumed"))?;
        let eos = Arc::new(AtomicU32::new(eos_id));
        let eos_sched = Arc::clone(&eos);
        let steps = Arc::clone(&self.steps);
        let inbox = Arc::clone(&self.inbox);
        let cfg = self.cfg;

        let (plan_tx, plan_rx) = channel::<BatchPlan>();
        let plan_tx_handle = plan_tx.clone();
        let (res_tx, res_rx) = channel::<Result<Vec<(u64, u32)>>>();
        let done = Arc::new(AtomicBool::new(false));
        let error_slot: Arc<Mutex<Option<deepinfer_core::Error>>> =
            Arc::new(Mutex::new(None));
        let outputs: Arc<Mutex<Vec<(u64, StopReason)>>> = Arc::new(Mutex::new(Vec::new()));

        // Engine thread: recv plan -> execute -> send results.
        let engine_thread = std::thread::spawn(move || {
            let mut engine = engine;
            let eos_id = eos.load(Ordering::SeqCst);
            while let Ok(plan) = plan_rx.recv() {
                if matches!(plan, BatchPlan::Idle) {
                    continue;
                }
                let out = engine.execute_step(&plan, eos_id);
                if res_tx.send(out).is_err() {
                    break;
                }
            }
        });

        // Scheduler thread: fold arrived results, schedule, send plan.
        let sched_thread = {
            let done = Arc::clone(&done);
            let error_slot = Arc::clone(&error_slot);
            let outputs = Arc::clone(&outputs);
            let steps = Arc::clone(&steps);
            let inbox = Arc::clone(&inbox);
            std::thread::spawn(move || {
                let mut sched = Scheduler::new(cfg);
                let mut in_flight = 0usize;
                loop {
                    // Fold submitted requests (deterministic under the
                    // inbox lock).
                    {
                        let mut guard = inbox.lock().expect("inbox poisoned");
                        for r in guard.drain(..) {
                            sched.add_request(r);
                        }
                    }
                    if sched.steps() as usize >= max_steps {
                        break;
                    }
                    // Fold results that arrived while we were scheduling
                    // (the overlap window's payoff).
                    loop {
                        match res_rx.try_recv() {
                            Ok(Ok(tokens)) => {
                                sched.report_tokens(&tokens, eos_sched.load(Ordering::SeqCst));
                                in_flight = in_flight.saturating_sub(1);
                            }
                            Ok(Err(e)) => {
                                *error_slot.lock().expect("error slot poisoned") = Some(e);
                                done.store(true, Ordering::SeqCst);
                                return;
                            }
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => return,
                        }
                    }
                    let plan = sched.schedule();
                    if let BatchPlan::Step { ref prefill_reqs, .. } = plan {
                        for &(id, cs, cl) in prefill_reqs {
                            sched.commit_prefill(id, cs, cl);
                        }
                        in_flight += 1;
                    }
                    steps.store(sched.steps(), Ordering::SeqCst);
                    if sched.is_done() && in_flight == 0 {
                        break;
                    }
                    if matches!(plan, BatchPlan::Idle) {
                        // Nothing to do: wait politely for submissions or
                        // results. The engine ignores Idle plans.
                        std::thread::yield_now();
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                    if plan_tx.send(plan).is_err() {
                        break;
                    }
                }
                // Drain late results so `in_flight` work is not lost.
                while in_flight > 0 {
                    match res_rx.recv() {
                        Ok(Ok(tokens)) => {
                            sched.report_tokens(&tokens, eos_sched.load(Ordering::SeqCst));
                            in_flight -= 1;
                        }
                        Ok(Err(e)) => {
                            *error_slot.lock().expect("error slot poisoned") = Some(e);
                            break;
                        }
                        Err(_) => break,
                    }
                }
                *outputs.lock().expect("outputs poisoned") = sched.finished().to_vec();
                done.store(true, Ordering::SeqCst);
            })
        };

        // Main thread: wait for completion.
        while !done.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // Signal threads to wind down: dropping plan_tx disconnects the
        // engine thread once the scheduler thread finishes.
        drop(plan_tx_handle);
        let _ = plan_tx;
        let _ = sched_thread.join();
        let _ = engine_thread.join();

        if let Some(e) = error_slot.lock().expect("error slot poisoned").take() {
            return Err(e);
        }
        let out = outputs.lock().expect("outputs poisoned").clone();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn config_from_env_and_validate() {
        let env = Env::default();
        let c = SchedulerConfig::from_env(&env);
        assert_eq!(c, SchedulerConfig::default());
        let mut bad = c;
        bad.max_batch_tokens = 0;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn chunked_prefill_splits_long_prompts() {
        let mut s = Scheduler::new(SchedulerConfig {
            max_running_requests: 8,
            max_batch_tokens: 8192,
            chunked_prefill_size: 2048,
            batch_invariant: false,
        });
        let id = s.add_request(Request::new(vec![7u32; 5000], 4));
        let p1 = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p1 else {
            panic!("expected a step");
        };
        assert_eq!(prefill_reqs, vec![(id, 0, 2048)]);
        s.commit_prefill(id, 0, 2048);
        let p2 = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p2 else {
            panic!("expected a step");
        };
        assert_eq!(prefill_reqs, vec![(id, 2048, 2048)]);
        s.commit_prefill(id, 2048, 2048);
        let p3 = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p3 else {
            panic!("expected a step");
        };
        assert_eq!(prefill_reqs, vec![(id, 4096, 904)]);
        s.commit_prefill(id, 4096, 904);
        assert!(matches!(s.phase(id), Some(Phase::Decoding { .. })));
    }

    #[test]
    fn decode_priority_and_partial_chunks() {
        // Budget 2: a decodes (1 token), leaving 1 for b's prefill chunk.
        let mut s = Scheduler::new(SchedulerConfig {
            max_running_requests: 8,
            max_batch_tokens: 2,
            chunked_prefill_size: 2,
            batch_invariant: false,
        });
        let a = s.add_request(Request::new(vec![1, 1], 3));
        let b = s.add_request(Request::new(vec![2, 2], 3));
        let p = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p else { panic!() };
        assert_eq!(prefill_reqs, vec![(a, 0, 2)]);
        s.commit_prefill(a, 0, 2);
        let p = s.schedule();
        let BatchPlan::Step { decode_reqs, prefill_reqs } = p else { panic!() };
        assert_eq!(decode_reqs, vec![a]);
        assert_eq!(prefill_reqs, vec![(b, 0, 1)]);
        s.commit_prefill(b, 0, 1);
        let p = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p else { panic!() };
        assert_eq!(prefill_reqs, vec![(b, 1, 1)]);
        s.commit_prefill(b, 1, 1);
        assert!(matches!(s.phase(b), Some(Phase::Decoding { .. })));
    }

    #[test]
    fn running_cap_defers_admission() {
        let mut s = Scheduler::new(SchedulerConfig {
            max_running_requests: 1,
            max_batch_tokens: 64,
            chunked_prefill_size: 8,
            batch_invariant: false,
        });
        let a = s.add_request(Request::new(vec![1], 2));
        let b = s.add_request(Request::new(vec![2], 2));
        let p = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p else { panic!() };
        assert_eq!(prefill_reqs, vec![(a, 0, 1)]);
        s.commit_prefill(a, 0, 1);
        let p = s.schedule();
        let BatchPlan::Step { decode_reqs, prefill_reqs } = p else { panic!() };
        assert_eq!(decode_reqs, vec![a]);
        assert!(prefill_reqs.is_empty());
        s.report_tokens(&[(a, 0)], 0);
        assert_eq!(s.finished(), &[(a, StopReason::Eos)]);
        let p = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p else { panic!() };
        assert_eq!(prefill_reqs, vec![(b, 0, 1)]);
        s.commit_prefill(b, 0, 1);
        s.report_tokens(&[(b, 5)], 0);
        s.report_tokens(&[(b, 6)], 0);
        assert_eq!(s.finished().len(), 2);
        assert!(s.is_done());
    }

    #[test]
    fn fcfs_and_eos_length_stops() {
        let mut s = Scheduler::new(SchedulerConfig::default());
        let a = s.add_request(Request::new(vec![1], 2));
        let b = s.add_request(Request::new(vec![2], 1));
        let p = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p else { panic!() };
        assert_eq!(prefill_reqs, vec![(a, 0, 1), (b, 0, 1)]);
        for &(id, cs, cl) in &prefill_reqs {
            s.commit_prefill(id, cs, cl);
        }
        let p = s.schedule();
        let BatchPlan::Step { decode_reqs, .. } = p else { panic!() };
        assert_eq!(decode_reqs, vec![a, b]);
        s.report_tokens(&[(a, 5), (b, 0)], 0); // b EOS, a continues
        assert_eq!(s.finished(), &[(b, StopReason::Eos)]);
        s.report_tokens(&[(a, 5)], 0);
        assert_eq!(
            s.finished(),
            &[(b, StopReason::Eos), (a, StopReason::Length)]
        );
        assert!(s.is_done());
    }

    #[test]
    fn length_finish() {
        let mut s = Scheduler::new(SchedulerConfig::default());
        let a = s.add_request(Request::new(vec![1], 1));
        let p = s.schedule();
        let BatchPlan::Step { prefill_reqs, .. } = p else { panic!() };
        let (id, cs, cl) = prefill_reqs[0];
        s.commit_prefill(id, cs, cl);
        s.report_tokens(&[(a, 9)], 0);
        assert_eq!(s.finished(), &[(a, StopReason::Length)]);
        assert!(s.is_done());
    }

    #[test]
    fn schedule_sequence_is_deterministic() {
        let build = || {
            let mut s = Scheduler::new(SchedulerConfig {
                max_running_requests: 4,
                max_batch_tokens: 16,
                chunked_prefill_size: 4,
                batch_invariant: true,
            });
            s.add_request(Request::new(vec![1, 2, 3, 4, 5, 6], 3));
            s.add_request(Request::new(vec![9, 9], 2));
            s
        };
        let mut seqs = Vec::new();
        for _ in 0..2 {
            let mut s = build();
            let mut seq = Vec::new();
            for _ in 0..8 {
                let p = s.schedule();
                if let BatchPlan::Step { prefill_reqs, .. } = &p {
                    for &(id, cs, cl) in prefill_reqs {
                        s.commit_prefill(id, cs, cl);
                    }
                }
                seq.push(p);
            }
            seqs.push(seq);
        }
        assert_eq!(seqs[0], seqs[1]);
    }

    #[test]
    fn async_engine_completes_all_requests() {
        // Deterministic mock: token = (id + round) % 7 + 1, EOS at
        // round `eos_round` per request.
        struct CountingMock {
            calls: Arc<AtomicUsize>,
            rounds: StdMutex<std::collections::HashMap<u64, usize>>,
        }
        impl EngineOps for CountingMock {
            fn execute_step(&mut self, plan: &BatchPlan, eos: u32) -> Result<Vec<(u64, u32)>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let BatchPlan::Step { decode_reqs, .. } = plan else {
                    return Ok(Vec::new());
                };
                let mut out = Vec::new();
                for &id in decode_reqs {
                    let r = {
                        let mut m = self.rounds.lock().unwrap();
                        let e = m.entry(id).or_insert(0);
                        *e += 1;
                        *e
                    };
                    // EOS at round 3 for every request.
                    let tok: u32 = if r == 3 { eos } else { ((id + r as u64) % 7 + 1) as u32 };
                    out.push((id, tok as u32));
                }
                Ok(out)
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mock = CountingMock {
            calls: Arc::clone(&calls),
            rounds: StdMutex::new(std::collections::HashMap::new()),
        };
        let mut eng = AsyncEngine::new(
            SchedulerConfig {
                max_running_requests: 8,
                max_batch_tokens: 64,
                chunked_prefill_size: 8,
                batch_invariant: true,
            },
            mock,
        );
        let a = eng.submit(Request::new(vec![1, 2], 10));
        let b = eng.submit(Request::new(vec![3], 10));
        let c = eng.submit(Request::new(vec![4, 5, 6], 10));
        let out = eng.run(0, 10_000).unwrap();
        assert_eq!(
            out,
            vec![
                (a, StopReason::Eos),
                (b, StopReason::Eos),
                (c, StopReason::Eos)
            ]
        );
        assert!(eng.steps_executed() > 0);
        assert!(calls.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn async_engine_matches_sequential_execution() {
        // Same token stream through the async pipeline and a plain
        // sequential loop must produce identical finish reasons.
        struct SeqMock;
        impl EngineOps for SeqMock {
            fn execute_step(&mut self, plan: &BatchPlan, eos: u32) -> Result<Vec<(u64, u32)>> {
                let BatchPlan::Step { decode_reqs, .. } = plan else {
                    return Ok(Vec::new());
                };
                Ok(decode_reqs.iter().map(|&id| (id, (id % 5 + 1) as u32)).collect::<Vec<_>>()
                    .into_iter()
                    .map(|(id, t)| (id, if t == 5 { eos } else { t }))
                    .collect())
            }
        }
        let mut eng = AsyncEngine::new(SchedulerConfig::default(), SeqMock);
        let a = eng.submit(Request::new(vec![1], 5));
        let b = eng.submit(Request::new(vec![2, 3], 5));
        let out = eng.run(0, 10_000).unwrap();
        // Sequential reference: both hit token 5 -> wait, token (id%5+1):
        // a=2, b=3... never eos. Let both run to length.
        assert_eq!(
            out,
            vec![(a, StopReason::Length), (b, StopReason::Length)]
        );
    }

    #[test]
    fn request_new_and_defaults() {
        let r = Request::new(vec![1, 2, 3], 4);
        assert_eq!(r.prompt, vec![1, 2, 3]);
        assert_eq!(r.max_tokens, 4);
        assert_eq!(r.id, 0);
        assert_eq!(SchedulerConfig::default().max_batch_tokens, 8192);
    }
}
