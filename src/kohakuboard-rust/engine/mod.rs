//! Bounded async logging engine owning the Rust writer thread and optional `SystemMonitor`.

pub mod events;
#[cfg(feature = "python")]
pub mod python;
pub mod snapshot;
#[cfg(test)]
mod tests;
pub mod worker;

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use parking_lot::{Mutex, RwLock};

use crate::error::{BoardError, BoardResult};
use crate::schema::{RunInfo, RunState};
use crate::storage::RunOpenOptions;
use crate::sysmon::SystemMonitor;

pub use events::{LogEvent, OverflowPolicy};
pub use snapshot::LiveSnapshot;

/// Configuration for launching a [`LoggerEngine`].
#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub open: RunOpenOptions,
    pub flush_interval_ms: u64,
    pub queue_capacity: usize,
    pub overflow: OverflowPolicy,
    pub system_metrics_interval_ms: Option<u64>,
}

/// Non-blocking logger handle forwarding events into the dedicated Rust writer thread.
pub struct LoggerEngine {
    tx: Mutex<Option<Sender<LogEvent>>>,
    rx_evict: Receiver<LogEvent>,
    overflow: OverflowPolicy,
    worker: Mutex<Option<JoinHandle<()>>>,
    sysmon: Mutex<Option<SystemMonitor>>,
    step_counter: Arc<AtomicI64>,
    global_step_counter: Arc<AtomicI64>,
    snapshot: Arc<RwLock<LiveSnapshot>>,
    dropped_events: Arc<AtomicU64>,
    closed: AtomicBool,
    initial_info: RunInfo,
}

impl LoggerEngine {
    /// Start the writer thread (and optional system metrics monitor) synchronously verifying storage init.
    pub fn start(cfg: EngineConfig) -> BoardResult<Self> {
        let cap = cfg.queue_capacity.max(16);
        let (tx, rx) = bounded::<LogEvent>(cap);
        let rx_evict = rx.clone();
        let (init_tx, init_rx) = bounded(1);

        let placeholder_info = RunInfo {
            schema_version: crate::schema::SCHEMA_VERSION,
            run_id: cfg.open.run_id.clone(),
            name: cfg.open.name.clone(),
            project: cfg.open.project.clone(),
            annotation: cfg.open.annotation.clone(),
            group: cfg.open.group.clone(),
            job_type: cfg.open.job_type.clone(),
            tags: cfg.open.tags.clone(),
            notes: cfg.open.notes.clone(),
            state: RunState::Running,
            created_at_ms: 0,
            updated_at_ms: 0,
            finished_at_ms: None,
            exit_code: None,
            pid: Some(std::process::id()),
            hostname: String::new(),
            last_step: -1,
            last_global_step: 0,
        };
        let snapshot = Arc::new(RwLock::new(LiveSnapshot::from_info(&placeholder_info)));
        let dropped_events = Arc::new(AtomicU64::new(0));
        let flush_interval = Duration::from_millis(cfg.flush_interval_ms.clamp(25, 60_000));

        let snap_bg = Arc::clone(&snapshot);
        let drop_bg = Arc::clone(&dropped_events);
        let open_opts = cfg.open;
        let handle = thread::Builder::new()
            .name("kobo-writer".into())
            .spawn(move || {
                worker::run_worker(open_opts, rx, flush_interval, snap_bg, drop_bg, init_tx);
            })
            .map_err(|e| BoardError::Storage(format!("failed to spawn writer thread: {e}")))?;

        let init = init_rx
            .recv()
            .map_err(|_| BoardError::Storage("writer thread terminated during init".into()))??;

        let step_counter = Arc::new(AtomicI64::new(init.info.last_step.max(0)));
        let global_step_counter = Arc::new(AtomicI64::new(init.info.last_global_step.max(0)));

        let sysmon = cfg.system_metrics_interval_ms.map(|ms| {
            SystemMonitor::start(
                ms,
                tx.clone(),
                Arc::clone(&step_counter),
                Arc::clone(&global_step_counter),
            )
        });

        Ok(Self {
            tx: Mutex::new(Some(tx)),
            rx_evict,
            overflow: cfg.overflow,
            worker: Mutex::new(Some(handle)),
            sysmon: Mutex::new(sysmon),
            step_counter,
            global_step_counter,
            snapshot,
            dropped_events,
            closed: AtomicBool::new(false),
            initial_info: init.info,
        })
    }

    /// Return the initial `RunInfo` loaded or created at startup (including resumed step counters).
    pub fn initial_info(&self) -> &RunInfo {
        &self.initial_info
    }

    /// True if `finish()` has already been called.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Update the shared atomic step counters used by the background system metrics sampler.
    pub fn set_step_counters(&self, step: i64, global_step: i64) {
        self.step_counter.store(step, Ordering::Relaxed);
        self.global_step_counter
            .store(global_step, Ordering::Relaxed);
    }

    /// Enqueue a non-control event respecting the configured overflow policy.
    pub fn send(&self, mut event: LogEvent) -> BoardResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(BoardError::Closed);
        }
        let sender = {
            let guard = self.tx.lock();
            guard.clone().ok_or(BoardError::Closed)?
        };
        match self.overflow {
            OverflowPolicy::Block => sender.send(event).map_err(|_| BoardError::Closed),
            OverflowPolicy::DropOldest => loop {
                match sender.try_send(event) {
                    Ok(()) => return Ok(()),
                    Err(TrySendError::Disconnected(_)) => return Err(BoardError::Closed),
                    Err(TrySendError::Full(ev)) => {
                        event = ev;
                        if let Ok(evicted) = self.rx_evict.try_recv() {
                            if evicted.is_control() {
                                // Never drop control barriers; block briefly to preserve them.
                                let _ = sender.send(evicted);
                            } else {
                                self.dropped_events.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            },
        }
    }

    /// Block until all events enqueued prior to this call are committed to SQLite.
    pub fn flush(&self, timeout_ms: u64) -> BoardResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(BoardError::Closed);
        }
        let sender = {
            let guard = self.tx.lock();
            guard.clone().ok_or(BoardError::Closed)?
        };
        let (ack_tx, ack_rx) = bounded(1);
        sender
            .send(LogEvent::Flush(ack_tx))
            .map_err(|_| BoardError::Closed)?;
        match ack_rx.recv_timeout(Duration::from_millis(timeout_ms.max(1))) {
            Ok(res) => res,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                Err(BoardError::FlushTimeout(timeout_ms))
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => Err(BoardError::Closed),
        }
    }

    /// Idempotently stop sysmon, drain the queue, checkpoint WAL, and join the writer thread.
    pub fn finish(&self, state: RunState, exit_code: i32, timeout_ms: u64) -> BoardResult<()> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        if let Some(mut sys) = self.sysmon.lock().take() {
            sys.stop();
        }
        let sender = self.tx.lock().take();
        let mut ack_res = Ok(());
        if let Some(tx) = sender {
            let (ack_tx, ack_rx) = bounded(1);
            if tx
                .send(LogEvent::Shutdown { state, exit_code, ack: ack_tx })
                .is_ok()
            {
                ack_res = match ack_rx.recv_timeout(Duration::from_millis(timeout_ms.max(1))) {
                    Ok(res) => res,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        Err(BoardError::FlushTimeout(timeout_ms))
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => Ok(()),
                };
            }
        }
        if let Some(handle) = self.worker.lock().take() {
            let _ = handle.join();
        }
        ack_res
    }

    /// Serialize the current in-memory live snapshot to JSON for TUI / observability consumers.
    pub fn snapshot_json(&self) -> BoardResult<String> {
        let snap = self.snapshot.read();
        Ok(serde_json::to_string(&*snap)?)
    }

    /// Return the number of events dropped under `DropOldest`.
    pub fn dropped_events(&self) -> u64 {
        self.dropped_events.load(Ordering::Relaxed)
    }
}

impl Drop for LoggerEngine {
    fn drop(&mut self) {
        let _ = self.finish(RunState::Finished, 0, 5_000);
    }
}
