//! Threading utilities: a fixed thread pool with deterministic
//! disjoint-range parallelism, plus a global default pool.
//!
//! The scheduling layer's "zero-overhead asynchronous" pipeline
//! (overlapping CPU batch scheduling with execution) is built on top of
//! these primitives with `std::sync::mpsc` channels (lock-free fast paths
//! provided by the standard library) — deliberately dependency-free and
//! `unsafe`-free.

use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

/// A fixed-size worker pool executing boxed jobs.
pub struct ThreadPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    tx: Mutex<Sender<Job>>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    /// Creates a pool with `threads` workers (clamped to `[1, 64]`).
    pub fn new(threads: usize) -> ThreadPool {
        let threads = threads.clamp(1, 64);
        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..threads {
            let rx = Arc::clone(&rx);
            std::thread::spawn(move || loop {
                let job = {
                    let guard = rx.lock().expect("pool channel poisoned");
                    guard.recv()
                };
                match job {
                    Ok(job) => job(),
                    Err(_) => break, // Sender dropped: shut down.
                }
            });
        }
        ThreadPool {
            inner: Arc::new(PoolInner {
                tx: Mutex::new(tx),
            }),
        }
    }

    /// Submits a job for asynchronous execution.
    pub fn spawn<F: FnOnce() + Send + 'static>(&self, f: F) {
        self.inner
            .tx
            .lock()
            .expect("pool channel poisoned")
            .send(Box::new(f))
            .expect("pool worker died");
    }

    /// Runs `f(start, end)` over `[0, total)` in disjoint chunks across
    /// workers and blocks until every chunk has completed.
    ///
    /// Chunks are disjoint, so results are independent of thread
    /// interleaving — the basis of deterministic parallel execution
    /// (batch invariance).
    pub fn par_for<F>(&self, total: usize, min_chunk: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        if total == 0 {
            return;
        }
        let min_chunk = min_chunk.max(1);
        let n_chunks = total.div_ceil(min_chunk);
        let chunk = total.div_ceil(n_chunks.max(1));
        let f = Arc::new(f);
        let (done_tx, done_rx) = channel::<()>();
        let mut spawned = 0usize;
        let mut start = 0usize;
        while start < total {
            let end = (start + chunk).min(total);
            let f = Arc::clone(&f);
            let done_tx = done_tx.clone();
            self.spawn(move || {
                f(start, end);
                let _ = done_tx.send(());
            });
            spawned += 1;
            start = end;
        }
        for _ in 0..spawned {
            let _ = done_rx.recv();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Dropping the sender closes the channel and terminates workers.
    }
}

static GLOBAL: std::sync::OnceLock<ThreadPool> = std::sync::OnceLock::new();

/// The process-wide default pool (`available_parallelism` workers,
/// clamped to `[1, 8]` to bound memory pressure in constrained
/// environments).
pub fn global_pool() -> &'static ThreadPool {
    GLOBAL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .clamp(1, 8);
        ThreadPool::new(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_for_covers_all_indices_exactly_once() {
        let pool = ThreadPool::new(4);
        let hits = Arc::new(std::sync::Mutex::new(vec![0u32; 1000]));
        let h = Arc::clone(&hits);
        pool.par_for(1000, 37, move |s, e| {
            let mut g = h.lock().unwrap();
            for i in s..e {
                g[i] += 1;
            }
        });
        assert!(hits.lock().unwrap().iter().all(|&c| c == 1));
    }

    #[test]
    fn par_for_disjoint_and_deterministic_sums() {
        let pool = ThreadPool::new(3);
        let data: Vec<u32> = (0..5000).map(|i| i as u32).collect();
        let data = Arc::new(data);
        let out = Arc::new(std::sync::Mutex::new(Vec::<(usize, usize, u64)>::new()));
        let o = Arc::clone(&out);
        let d = Arc::clone(&data);
        pool.par_for(d.len(), 128, move |s, e| {
            let sum: u64 = d[s..e].iter().map(|v| *v as u64).sum();
            o.lock().unwrap().push((s, e, sum));
        });
        let mut chunks = out.lock().unwrap().clone();
        chunks.sort();
        let mut expect = 0usize;
        for (s, e, _) in &chunks {
            assert_eq!(*s, expect);
            expect = *e;
        }
        assert_eq!(expect, data.len());
    }

    #[test]
    fn zero_and_small_totals() {
        let pool = ThreadPool::new(2);
        pool.par_for(0, 8, |_, _| panic!("should not run"));
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let r = Arc::clone(&ran);
        pool.par_for(3, 8, move |s, e| {
            assert_eq!(s, 0);
            assert_eq!(e, 3);
            r.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        assert!(ran.load(std::sync::atomic::Ordering::SeqCst));
    }
}
