//! Single-writer background OS thread processing batched events off-GIL.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use parking_lot::RwLock;

use crate::compute::histogram::{compute_histogram, pack_precomputed};
use crate::compute::kde::compute_kde;
use crate::compute::media::{
    blake3_short_hex, compress_bytes, encode_image_pixels, wrap_encoded_media,
};
use crate::error::BoardResult;
use crate::schema::{RunInfo, RunState, ScalarRow};
use crate::storage::{RunOpenOptions, RunStore};

use super::events::LogEvent;
use super::snapshot::LiveSnapshot;

const MAX_BURST_EVENTS: usize = 4_096;

/// Startup payload returned synchronously from the worker thread to `LoggerEngine::start`.
pub struct WorkerInit {
    pub info: RunInfo,
}

/// Entrypoint for the single-writer background thread.
pub fn run_worker(
    open_opts: RunOpenOptions,
    rx: Receiver<LogEvent>,
    flush_interval: Duration,
    snapshot: Arc<RwLock<LiveSnapshot>>,
    dropped_counter: Arc<AtomicU64>,
    init_tx: Sender<BoardResult<WorkerInit>>,
) {
    let mut store = match RunStore::open(open_opts) {
        Ok(s) => {
            {
                let mut snap = snapshot.write();
                *snap = LiveSnapshot::from_info(&s.info);
            }
            let _ = init_tx.send(Ok(WorkerInit { info: s.info.clone() }));
            s
        }
        Err(err) => {
            let _ = init_tx.send(Err(err));
            return;
        }
    };

    let mut scalar_buf: BTreeMap<String, Vec<ScalarRow>> = BTreeMap::new();
    let mut log_buf: Vec<(i64, i64, String, String)> = Vec::new();
    let mut last_flush = Instant::now();

    loop {
        let first = match rx.recv_timeout(flush_interval) {
            Ok(ev) => Some(ev),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => {
                // All senders dropped without explicit Shutdown — flush and finalize cleanly.
                let _ = flush_buffers(&mut store, &mut scalar_buf, &mut log_buf);
                let _ = store.finalize(RunState::Finished, 0);
                {
                    let mut snap = snapshot.write();
                    snap.state = RunState::Finished;
                    snap.dropped_events = dropped_counter.load(Ordering::Relaxed);
                }
                store.close();
                return;
            }
        };

        let mut shutdown_req: Option<(RunState, i32, Sender<BoardResult<()>>)> = None;
        let mut flush_acks: Vec<Sender<BoardResult<()>>> = Vec::new();

        if let Some(ev) = first {
            handle_event(
                ev,
                &mut store,
                &mut scalar_buf,
                &mut log_buf,
                &snapshot,
                &mut flush_acks,
                &mut shutdown_req,
            );

            // Burst-drain immediately ready events up to MAX_BURST_EVENTS.
            for _ in 0..MAX_BURST_EVENTS {
                if shutdown_req.is_some() {
                    break;
                }
                match rx.try_recv() {
                    Ok(ev) => handle_event(
                        ev,
                        &mut store,
                        &mut scalar_buf,
                        &mut log_buf,
                        &snapshot,
                        &mut flush_acks,
                        &mut shutdown_req,
                    ),
                    Err(_) => break,
                }
            }
        }

        if let Some((state, exit_code, ack)) = shutdown_req {
            // Drain any remaining events already queued before Shutdown.
            while let Ok(ev) = rx.try_recv() {
                let mut ignored_shutdown = None;
                handle_event(
                    ev,
                    &mut store,
                    &mut scalar_buf,
                    &mut log_buf,
                    &snapshot,
                    &mut flush_acks,
                    &mut ignored_shutdown,
                );
            }
            let res = flush_buffers(&mut store, &mut scalar_buf, &mut log_buf)
                .and_then(|()| store.finalize(state, exit_code));
            for f_ack in flush_acks.drain(..) {
                let _ = f_ack.send(Ok(()));
            }
            {
                let mut snap = snapshot.write();
                snap.state = state;
                snap.dropped_events = dropped_counter.load(Ordering::Relaxed);
            }
            store.close();
            let _ = ack.send(res);
            return;
        }

        let need_flush = !flush_acks.is_empty()
            || last_flush.elapsed() >= flush_interval
            || scalar_buf.len() >= 128;

        if need_flush {
            let res = flush_buffers(&mut store, &mut scalar_buf, &mut log_buf);
            last_flush = Instant::now();
            {
                let mut snap = snapshot.write();
                snap.dropped_events = dropped_counter.load(Ordering::Relaxed);
            }
            if let Err(err) = res {
                let msg = err.to_string();
                for ack in flush_acks.drain(..) {
                    let _ = ack.send(Err(crate::error::BoardError::Storage(msg.clone())));
                }
            } else {
                for ack in flush_acks.drain(..) {
                    let _ = ack.send(Ok(()));
                }
            }
        }
    }
}

fn flush_buffers(
    store: &mut RunStore,
    scalar_buf: &mut BTreeMap<String, Vec<ScalarRow>>,
    log_buf: &mut Vec<(i64, i64, String, String)>,
) -> BoardResult<()> {
    if !scalar_buf.is_empty() {
        store.append_scalar_batches(scalar_buf)?;
        scalar_buf.clear();
    }
    if !log_buf.is_empty() {
        store.insert_log_lines(log_buf)?;
        log_buf.clear();
    }
    store.flush_metadata()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_event(
    event: LogEvent,
    store: &mut RunStore,
    scalar_buf: &mut BTreeMap<String, Vec<ScalarRow>>,
    log_buf: &mut Vec<(i64, i64, String, String)>,
    snapshot: &Arc<RwLock<LiveSnapshot>>,
    flush_acks: &mut Vec<Sender<BoardResult<()>>>,
    shutdown_req: &mut Option<(RunState, i32, Sender<BoardResult<()>>)>,
) {
    match event {
        LogEvent::Scalars { step, global_step, ts_ms, metrics } => {
            let mut snap = snapshot.write();
            for (name, value) in metrics {
                snap.record_scalar(&name, step, global_step, ts_ms, value);
                scalar_buf.entry(name).or_default().push(ScalarRow {
                    step,
                    global_step,
                    ts_ms,
                    value,
                });
            }
        }
        LogEvent::MediaPixels {
            step,
            global_step,
            ts_ms,
            name,
            idx,
            pixels,
            width,
            height,
            channels,
            format,
            quality,
            caption,
            meta_json,
        } => {
            if let Ok(enc) = encode_image_pixels(&pixels, width, height, channels, &format, quality)
            {
                let size_bytes = enc.bytes.len() as i64;
                if store.put_blob_dedup(&enc.blob_key, &enc.bytes).is_ok() {
                    let _ = store.insert_media_event(
                        step,
                        global_step,
                        ts_ms,
                        &name,
                        idx,
                        "image",
                        &enc.blob_key,
                        &enc.mime,
                        size_bytes,
                        enc.width,
                        enc.height,
                        None,
                        &caption,
                        &meta_json,
                    );
                }
            }
        }
        LogEvent::MediaEncoded {
            step,
            global_step,
            ts_ms,
            name,
            idx,
            media_type,
            bytes,
            ext,
            duration_ms,
            caption,
            meta_json,
        } => {
            let enc = wrap_encoded_media(bytes, &ext, &media_type);
            let size_bytes = enc.bytes.len() as i64;
            if store.put_blob_dedup(&enc.blob_key, &enc.bytes).is_ok() {
                let _ = store.insert_media_event(
                    step,
                    global_step,
                    ts_ms,
                    &name,
                    idx,
                    &media_type,
                    &enc.blob_key,
                    &enc.mime,
                    size_bytes,
                    enc.width,
                    enc.height,
                    duration_ms,
                    &caption,
                    &meta_json,
                );
            }
        }
        LogEvent::PutBlob { blob_key, bytes } => {
            let _ = store.put_blob_dedup(&blob_key, &bytes);
        }
        LogEvent::Table { step, global_step, ts_ms, name, columns_json, row_count, rows_json } => {
            let _ = store.insert_table(
                step,
                global_step,
                ts_ms,
                &name,
                &columns_json,
                row_count,
                &rows_json,
            );
        }
        LogEvent::HistogramRaw {
            step,
            global_step,
            ts_ms,
            name,
            values,
            bins,
            precision,
            range,
            percentiles,
        } => {
            if let Ok(Some(h)) = compute_histogram(&values, bins, precision, range, percentiles) {
                let _ = store.append_histogram(
                    &name,
                    step,
                    global_step,
                    ts_ms,
                    h.min_val,
                    h.max_val,
                    h.bins,
                    h.precision,
                    &h.counts_bytes,
                );
            }
        }
        LogEvent::HistogramPrecomputed {
            step,
            global_step,
            ts_ms,
            name,
            edges,
            counts,
            precision,
        } => {
            if let Ok(h) = pack_precomputed(&edges, &counts, precision) {
                let _ = store.append_histogram(
                    &name,
                    step,
                    global_step,
                    ts_ms,
                    h.min_val,
                    h.max_val,
                    h.bins,
                    h.precision,
                    &h.counts_bytes,
                );
            }
        }
        LogEvent::KdeRaw {
            step,
            global_step,
            ts_ms,
            name,
            values,
            num_points,
            bandwidth,
            approximate,
            range,
            percentiles,
        } => {
            if let Ok(Some(k)) =
                compute_kde(&values, num_points, bandwidth, approximate, range, percentiles)
            {
                let _ = store.append_kde(
                    &name,
                    step,
                    global_step,
                    ts_ms,
                    k.range_min,
                    k.range_max,
                    k.bandwidth,
                    k.sample_count,
                    &k.grid,
                    &k.density,
                );
            }
        }
        LogEvent::KdePrecomputed {
            step,
            global_step,
            ts_ms,
            name,
            range_min,
            range_max,
            bandwidth,
            sample_count,
            grid,
            density,
        } => {
            let _ = store.append_kde(
                &name,
                step,
                global_step,
                ts_ms,
                range_min,
                range_max,
                bandwidth,
                sample_count,
                &grid,
                &density,
            );
        }
        LogEvent::Tensor {
            step,
            global_step,
            ts_ms,
            name,
            raw_bytes,
            dtype,
            shape_json,
            compress,
            meta_json,
        } => {
            let (payload, is_compressed, ext) = if compress {
                match compress_bytes(&raw_bytes) {
                    Ok(c) => (c, true, "tensor.gz"),
                    Err(_) => (raw_bytes, false, "tensor"),
                }
            } else {
                (raw_bytes, false, "tensor")
            };
            let hash = blake3_short_hex(&payload);
            let blob_key = format!("{hash}.{ext}");
            let size_bytes = payload.len() as i64;
            if store.put_blob_dedup(&blob_key, &payload).is_ok() {
                let _ = store.insert_tensor(
                    step,
                    global_step,
                    ts_ms,
                    &name,
                    &blob_key,
                    &dtype,
                    &shape_json,
                    size_bytes,
                    is_compressed,
                    &meta_json,
                );
            }
        }
        LogEvent::Artifact {
            step,
            global_step,
            ts_ms,
            name,
            kind,
            uri,
            digest,
            size_bytes,
            meta_json,
        } => {
            let _ = store.insert_artifact(
                step,
                global_step,
                ts_ms,
                &name,
                &kind,
                &uri,
                digest.as_deref(),
                size_bytes,
                &meta_json,
            );
        }
        LogEvent::Alert { step, global_step, ts_ms, level, title, text } => {
            let _ = store.insert_alert(step, global_step, ts_ms, &level, &title, &text);
            snapshot
                .write()
                .record_alert(step, ts_ms, level, title, text);
        }
        LogEvent::LogLines { mut lines } => {
            log_buf.append(&mut lines);
            if log_buf.len() >= 128 {
                let _ = store.insert_log_lines(log_buf);
                log_buf.clear();
            }
        }
        LogEvent::ConfigPatch { patch } => {
            store.patch_config(&patch);
        }
        LogEvent::SummaryPatch { step, global_step, ts_ms, values } => {
            store.patch_summary(&values, step, global_step, ts_ms);
        }
        LogEvent::MetaPatch { name, group, add_tags, notes } => {
            store.patch_meta(name.clone(), group, add_tags, notes);
            if let Some(n) = name {
                snapshot.write().name = n;
            }
        }
        LogEvent::DefineMetric(def) => {
            store.define_metric(def);
        }
        LogEvent::Progress { label, current, total, unit, ts_ms } => {
            snapshot
                .write()
                .record_progress(label, current, total, unit, ts_ms);
        }
        LogEvent::Flush(ack) => {
            flush_acks.push(ack);
        }
        LogEvent::Shutdown { state, exit_code, ack } => {
            *shutdown_req = Some((state, exit_code, ack));
        }
    }
}
