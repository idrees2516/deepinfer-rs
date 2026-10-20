//! Deferred-completion hook — the DeepEP `EventOverlap` / `return_recv_hook`
//! analogue.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

/// Deferred-completion handle mirroring DeepEP's `EventOverlap` /
/// `return_recv_hook` objects.
///
/// In this in-process simulation every operation completes synchronously,
/// so hooks are constructed **already fulfilled**; [`EpHook::wait`] returns
/// immediately and [`EpHook::is_fulfilled`] is always `true`. The type
/// keeps the upstream structure — an atomic fulfilled flag plus a
/// (mutex, condvar) waker channel — so a future async transport (real
/// NVLink/RDMA queues, background copy engines) can flip the flag and
/// wake waiters without changing the API. Lock poisoning is recovered from
/// rather than panicking (this crate never panics).
#[derive(Debug)]
pub struct EpHook {
    /// Whether the tracked operation has completed.
    fulfilled: AtomicBool,
    /// Waker channel guarding `fulfilled` transitions (condvar pair).
    signal: Mutex<()>,
    /// Notified when `fulfilled` becomes true.
    cond: Condvar,
}

impl EpHook {
    /// Creates an already-fulfilled hook (this simulation completes every
    /// operation synchronously). Auxiliary constructor: DeepEP creates its
    /// hooks internally around `EventOverlap` records.
    pub fn new() -> Self {
        EpHook {
            fulfilled: AtomicBool::new(true),
            signal: Mutex::new(()),
            cond: Condvar::new(),
        }
    }

    /// Blocks until the hook is fulfilled (mirrors DeepEP's
    /// `EventOverlap.current_event_wait` / `hook.wait()`). Returns
    /// immediately for synchronously-fulfilled hooks.
    ///
    /// Never panics: a poisoned lock is recovered from, and the wait loop
    /// re-checks the flag on every wakeup.
    pub fn wait(&self) {
        let mut guard = self.signal.lock().unwrap_or_else(|e| e.into_inner());
        while !self.fulfilled.load(Ordering::Acquire) {
            guard = self.cond.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Whether the tracked operation has completed.
    pub fn is_fulfilled(&self) -> bool {
        self.fulfilled.load(Ordering::Acquire)
    }
}

impl Default for EpHook {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_is_immediately_fulfilled() {
        let hook = EpHook::new();
        assert!(hook.is_fulfilled());
        hook.wait(); // must return immediately
        assert!(hook.is_fulfilled());
        assert!(EpHook::default().is_fulfilled());
    }
}
