/**
 * REST, newest-first binary chunk streaming, and SSE client for KohakuBoard Web API.
 */

import { consumeChunkStream, decodeSingleFrame } from "./binaryStream"

function encPath(segment) {
  return String(segment)
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/")
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const err = await res.json()
      if (err && err.detail) msg = err.detail
    } catch {
      // ignore non-json error body
    }
    throw new Error(msg)
  }
  return res.json()
}

export const api = {
  systemInfo: () => fetchJson("/api/system/info"),
  listProjects: () => fetchJson("/api/projects"),

  listRuns: (project, params = {}) => {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined && v !== "") qs.set(k, String(v))
    }
    const q = qs.toString()
    return fetchJson(`/api/projects/${encodeURIComponent(project)}/runs${q ? `?${q}` : ""}`)
  },

  getRun: (project, runId) =>
    fetchJson(`/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}`),

  batchRunMeta: (project, runIds) =>
    fetchJson(`/api/projects/${encodeURIComponent(project)}/runs/batch/meta`, {
      method: "POST",
      body: JSON.stringify({ run_ids: runIds }),
    }),

  patchRun: (project, runId, patch) =>
    fetchJson(`/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  deleteRun: (project, runId) =>
    fetchJson(`/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}`, {
      method: "DELETE",
    }),

  streamScalars: async (
    project,
    runIds,
    metrics,
    onFrame,
    { firstChunk = 8192, maxChunk = 65536, signal } = {},
  ) => {
    const qs = new URLSearchParams({
      runs: runIds.join(","),
      metrics: metrics.join(","),
      x_dtype: "f64",
      first_chunk: String(firstChunk),
      max_chunk: String(maxChunk),
    })
    const res = await fetch(
      `/api/projects/${encodeURIComponent(project)}/runs/scalars/stream?${qs.toString()}`,
      { signal },
    )
    if (!res.ok) throw new Error(`Stream failed: ${res.status}`)
    await consumeChunkStream(res, onFrame)
  },

  scalarTailPage: async (project, runId, metric, after, limit = 16384, signal) => {
    const qs = new URLSearchParams({
      after: String(after),
      limit: String(limit),
      x_dtype: "f64",
    })
    const res = await fetch(
      `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/scalars/${encPath(metric)}/page?${qs.toString()}`,
      { signal },
    )
    if (!res.ok) throw new Error(`Tail page failed: ${res.status}`)
    return decodeSingleFrame(await res.arrayBuffer())
  },

  mediaEvents: (project, runId, name, { beforeId = null, limit = 24 } = {}) => {
    const qs = new URLSearchParams({ limit: String(limit) })
    if (beforeId !== null) qs.set("before_id", String(beforeId))
    return fetchJson(
      `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/media/${encPath(name)}?${qs.toString()}`,
    )
  },

  blobUrl: (project, runId, key) =>
    `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/blobs/${encodeURIComponent(key)}`,

  tables: (project, runId, name, { beforeId = null, limit = 10 } = {}) => {
    const qs = new URLSearchParams({ limit: String(limit) })
    if (beforeId !== null) qs.set("before_id", String(beforeId))
    return fetchJson(
      `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/tables/${encPath(name)}?${qs.toString()}`,
    )
  },

  distributionsPage: (project, runId, kind, name, { before = null, limit = 32 } = {}) => {
    const qs = new URLSearchParams({ limit: String(limit) })
    if (before !== null) qs.set("before", String(before))
    return fetchJson(
      `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/distributions/${encodeURIComponent(kind)}/${encPath(name)}/page?${qs.toString()}`,
    )
  },

  artifacts: (project, runId, limit = 50) =>
    fetchJson(
      `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/artifacts?limit=${limit}`,
    ),

  alerts: (project, runId, limit = 50) =>
    fetchJson(
      `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/alerts?limit=${limit}`,
    ),

  logsTail: (project, runId, { beforeId = null, limit = 200 } = {}) => {
    const qs = new URLSearchParams({ limit: String(limit) })
    if (beforeId !== null) qs.set("before_id", String(beforeId))
    return fetchJson(
      `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/logs?${qs.toString()}`,
    )
  },

  logsSearch: (project, runId, query, limit = 50) => {
    const qs = new URLSearchParams({ q: query, limit: String(limit) })
    return fetchJson(
      `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/logs/search?${qs.toString()}`,
    )
  },

  subscribeProjectEvents: (project, onMessage) => {
    const es = new EventSource(`/api/projects/${encodeURIComponent(project)}/events`)
    es.addEventListener("runs", (ev) => {
      try {
        onMessage(JSON.parse(ev.data))
      } catch {
        // ignore malformed event
      }
    })
    return es
  },
}
