//! Bounded-tail heterogeneous chunk queue: E-core helpers for flat passes.
//!
//! Folding E-cores into the main rayon pool is a known regression on Apple
//! silicon: kernels that hand each worker one equal band gate their barrier
//! on the slowest core, and dynamic work-stealing still parks whole stolen
//! subtrees on E-cores (measured here: blanket 10 threads costs the
//! zerocheck +14 ms and lincheck +6 ms at m=32). This module uses the
//! opposite primitive, re-derived from the challenge tree's epool: the main
//! (P-core) pool and a tiny E-core-only helper pool drain ONE shared atomic
//! index. Nobody owns a band; an E-core only ever holds the single chunk it
//! most recently claimed, so when the P-cores exhaust the queue the barrier
//! waits on at most one E-sized chunk per helper — the straggler exposure is
//! bounded by construction instead of by luck.
//!
//! Use it for passes shaped like "N independent chunks, each writing a
//! disjoint output range and returning a small value merged by an exact
//! commutative operation" (the zerocheck's round-2 fused fold and rounds-3+
//! tail). Because F_2-characteristic addition is exact and order-free, the
//! output is bit-identical to the rayon path regardless of which core ran
//! which chunk.
//!
//! `FLOCK_NO_EPOOL=1` is the kill switch (read once); on non-macOS, non-
//! aarch64, or E-core-less machines the helpers never exist and
//! [`hetero_enabled`] is false — call sites keep their rayon path for that
//! case, so disabling restores the incumbent scheduling exactly.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// `*mut F128`-style pointer that call sites may smuggle into the chunk
/// closure for disjoint-range writes. The `get` accessor forces whole-struct
/// closure capture (a raw-pointer FIELD capture is not `Sync`).
pub struct SyncMutPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}
impl<T> SyncMutPtr<T> {
    #[inline(always)]
    pub fn get(&self) -> *mut T {
        self.0
    }
}

/// Number of efficiency cores the helper pool may use (0 disables).
fn e_core_count() -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        let total = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        total.saturating_sub(crate::perf_core_count_cached())
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        0
    }
}

/// Tag the current thread background-QoS: macOS then schedules it onto
/// efficiency cores. Same trick as the commit-phase prefault thread.
#[cfg(target_os = "macos")]
fn set_background_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // QOS_CLASS_BACKGROUND = 0x09.
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x09, 0);
    }
}

/// One type-erased job for the helper threads. The pointee lives on the
/// submitting caller's stack; validity is guaranteed by the submit/complete
/// protocol (the caller blocks until `remaining == 0`).
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn() + Sync));
unsafe impl Send for Job {}

struct Slot {
    epoch: u64,
    job: Option<Job>,
    /// Helpers that have not yet finished the current epoch's job.
    remaining: usize,
}

struct Shared {
    slot: Mutex<Slot>,
    start: Condvar,
    done: Condvar,
    n_helpers: usize,
}

fn shared() -> Option<&'static Shared> {
    static SHARED: OnceLock<Option<&'static Shared>> = OnceLock::new();
    *SHARED.get_or_init(|| {
        if std::env::var_os("FLOCK_NO_EPOOL").is_some() {
            return None;
        }
        let n = e_core_count();
        if n == 0 {
            return None;
        }
        let sh: &'static Shared = Box::leak(Box::new(Shared {
            slot: Mutex::new(Slot {
                epoch: 0,
                job: None,
                remaining: 0,
            }),
            start: Condvar::new(),
            done: Condvar::new(),
            n_helpers: n,
        }));
        for i in 0..n {
            std::thread::Builder::new()
                .name(format!("flock-epool-{i}"))
                .stack_size(8 << 20)
                .spawn(move || helper_main(sh))
                .expect("epool helper spawn");
        }
        Some(sh)
    })
}

fn helper_main(sh: &'static Shared) {
    #[cfg(target_os = "macos")]
    set_background_qos();
    let mut seen = 0u64;
    loop {
        let job = {
            let mut slot = sh.slot.lock().unwrap();
            loop {
                if slot.epoch > seen {
                    if let Some(job) = slot.job {
                        seen = slot.epoch;
                        break job;
                    }
                }
                slot = sh.start.wait(slot).unwrap();
            }
        };
        // SAFETY: the submitting caller blocks in `run_hetero_chunks` until
        // `remaining` hits zero, so the pointee outlives this call.
        (unsafe { &*job.0 })();
        let mut slot = sh.slot.lock().unwrap();
        slot.remaining -= 1;
        if slot.remaining == 0 {
            sh.done.notify_all();
        }
    }
}

/// Whether the E-core helper pool exists (E-cores present, kill switch off).
/// Call sites keep their incumbent rayon path when this is false, so
/// `FLOCK_NO_EPOOL=1` restores the old scheduling exactly.
pub fn hetero_enabled() -> bool {
    shared().is_some()
}

/// Drain chunks `0..n_chunks` through one shared atomic index, on every
/// thread of the CURRENT rayon pool plus the E-core helpers. `f(i)` computes
/// chunk `i` (doing its own disjoint-range output writes) and returns a
/// value; per-thread partial results are combined with `merge`, which must
/// be exact, commutative and associative (field additions are). Returns
/// `zero()` for an empty queue.
///
/// Returns only after every chunk — including any tail chunk an E-core
/// holds — has completed.
pub fn run_hetero_chunks<R, F, M, Z>(n_chunks: usize, f: F, zero: Z, merge: M) -> R
where
    R: Send,
    F: Fn(usize) -> R + Sync,
    M: Fn(R, R) -> R + Send + Sync,
    Z: FnOnce() -> R,
{
    let next = AtomicUsize::new(0);
    let result: Mutex<Option<R>> = Mutex::new(None);
    let worker = || {
        let mut acc: Option<R> = None;
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            let r = f(i);
            acc = Some(match acc {
                Some(a) => merge(a, r),
                None => r,
            });
        }
        if let Some(a) = acc {
            let mut g = result.lock().unwrap();
            let prev = g.take();
            *g = Some(match prev {
                Some(p) => merge(p, a),
                None => a,
            });
        }
    };

    // Arm the helpers (if any), then drain on the current pool's threads.
    let submitted = shared().map(|sh| {
        // SAFETY: lifetime erasure of a stack closure into the job slot. The
        // wait loop below does not return until every helper has finished
        // this epoch's job (`remaining == 0`), and the slot is cleared before
        // return — no helper can observe the pointer after `worker` dies.
        let job = Job(unsafe {
            core::mem::transmute::<&(dyn Fn() + Sync), &'static (dyn Fn() + Sync)>(
                &worker as &(dyn Fn() + Sync),
            ) as *const _
        });
        {
            let mut slot = sh.slot.lock().unwrap();
            slot.epoch += 1;
            slot.job = Some(job);
            slot.remaining = sh.n_helpers;
            sh.start.notify_all();
        }
        sh
    });
    rayon::broadcast(|_| worker());
    if let Some(sh) = submitted {
        let mut slot = sh.slot.lock().unwrap();
        while slot.remaining > 0 {
            slot = sh.done.wait(slot).unwrap();
        }
        slot.job = None;
    }

    result.into_inner().unwrap().unwrap_or_else(zero)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drains_every_chunk_once_and_merges() {
        // Sum of chunk indices via the queue == closed form; every index
        // consumed exactly once regardless of who ran it.
        for n in [0usize, 1, 7, 64, 1023] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let total = run_hetero_chunks(
                n,
                |i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                    i as u64
                },
                || 0u64,
                |a, b| a + b,
            );
            assert_eq!(total, (n as u64).saturating_sub(1) * n as u64 / 2);
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn nested_invocations_reuse_helpers() {
        // Two back-to-back jobs must not deadlock on the epoch protocol.
        for _ in 0..3 {
            let s = run_hetero_chunks(100, |i| i as u64, || 0, |a, b| a + b);
            assert_eq!(s, 4950);
        }
    }
}
