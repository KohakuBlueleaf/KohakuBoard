import { defineStore } from "pinia"
import { ref } from "vue"
import { api } from "@/utils/api"
import { SeriesBuffer } from "@/utils/seriesBuffer"

/**
 * Shared SeriesBuffer cache + newest-first progressive streaming scheduler.
 *
 * Multiple visible charts coalesce their requests into one round-robin
 * `/api/projects/{p}/runs/scalars/stream` call so every visible series paints
 * its newest 8,192 points immediately while older history streams backward to 0.
 */
export const useSeriesCacheStore = defineStore("seriesCache", () => {
  const buffers = new Map()
  const inFlight = new Set()
  const versions = ref({})
  const streamingCount = ref(0)

  let _pendingQueue = new Map() // project -> { runs: Set, metrics: Set }
  let _flushTimer = null
  let _rafPending = false
  const _dirtyKeys = new Set()

  function keyOf(project, runId, metric) {
    return `${project}/${runId}/${metric}`
  }

  function getBuffer(project, runId, metric) {
    return buffers.get(keyOf(project, runId, metric)) || null
  }

  function getVersion(project, runId, metric) {
    return versions.value[keyOf(project, runId, metric)] || 0
  }

  function _scheduleBump(key) {
    _dirtyKeys.add(key)
    if (_rafPending) return
    _rafPending = true
    const flush = () => {
      _rafPending = false
      if (_dirtyKeys.size === 0) return
      const next = { ...versions.value }
      for (const k of _dirtyKeys) {
        next[k] = (next[k] || 0) + 1
      }
      _dirtyKeys.clear()
      versions.value = next
    }
    if (typeof requestAnimationFrame !== "undefined") {
      requestAnimationFrame(flush)
    } else {
      setTimeout(flush, 16)
    }
  }

  function requestMetric(project, runIds, metric) {
    if (!project || !metric || !runIds || runIds.length === 0) return
    const neededRuns = []
    for (const rid of runIds) {
      const k = keyOf(project, rid, metric)
      const buf = buffers.get(k)
      if ((buf && buf.complete) || inFlight.has(k)) continue
      neededRuns.push(rid)
    }
    if (neededRuns.length === 0) return

    if (!_pendingQueue.has(project)) {
      _pendingQueue.set(project, { runs: new Set(), metrics: new Set() })
    }
    const bucket = _pendingQueue.get(project)
    for (const rid of neededRuns) bucket.runs.add(rid)
    bucket.metrics.add(metric)

    if (!_flushTimer) {
      _flushTimer = setTimeout(_dispatchBatch, 20)
    }
  }

  async function _dispatchBatch() {
    _flushTimer = null
    const snapshot = Array.from(_pendingQueue.entries())
    _pendingQueue = new Map()

    for (const [project, bucket] of snapshot) {
      const runIds = Array.from(bucket.runs)
      const metrics = Array.from(bucket.metrics)
      const keys = []
      for (const rid of runIds) {
        for (const m of metrics) {
          const k = keyOf(project, rid, m)
          if (!inFlight.has(k)) {
            inFlight.add(k)
            keys.push(k)
          }
        }
      }
      if (keys.length === 0) continue

      streamingCount.value++
      try {
        await api.streamScalars(
          project,
          runIds,
          metrics,
          (frame) => {
            const { header, step, globalStep, tsMs, value } = frame
            const k = keyOf(project, header.run_id, header.metric)
            let buf = buffers.get(k)
            if (!buf) {
              buf = new SeriesBuffer(header.total_rows)
              buffers.set(k, buf)
            }
            buf.writeChunk(
              header.start,
              header.end,
              step,
              globalStep,
              tsMs,
              value,
              header.total_rows,
            )
            _scheduleBump(k)
          },
          { firstChunk: 8192, maxChunk: 65536 },
        )
      } catch {
        // aborted or transient network error
      } finally {
        for (const k of keys) inFlight.delete(k)
        streamingCount.value = Math.max(0, streamingCount.value - 1)
      }
    }
  }

  async function pollLiveTail(project, runId) {
    const prefix = `${project}/${runId}/`
    for (const [k, buf] of buffers.entries()) {
      if (!k.startsWith(prefix)) continue
      const metric = k.slice(prefix.length)
      try {
        const frame = await api.scalarTailPage(project, runId, metric, buf.loadedEnd)
        if (frame && frame.header.end > frame.header.start) {
          buf.writeChunk(
            frame.header.start,
            frame.header.end,
            frame.step,
            frame.globalStep,
            frame.tsMs,
            frame.value,
            frame.header.total_rows,
          )
          _scheduleBump(k)
        }
      } catch {
        // ignore transient tail poll failure
      }
    }
  }

  return {
    versions,
    streamingCount,
    getBuffer,
    getVersion,
    requestMetric,
    pollLiveTail,
  }
})
