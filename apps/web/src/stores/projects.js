import { defineStore } from "pinia"
import { ref } from "vue"
import { api } from "@/utils/api"

export const useProjectsStore = defineStore("projects", () => {
  const systemInfo = ref(null)
  const projects = ref([])
  const runsByProject = ref({})
  const detailsByKey = ref({})
  const loadingProjects = ref(false)
  const error = ref(null)
  const sseConnected = ref(false)

  const _sseByProject = new Map()
  const _listeners = new Set()

  function onRunUpdates(cb) {
    _listeners.add(cb)
    return () => _listeners.delete(cb)
  }

  async function loadSystem() {
    try {
      systemInfo.value = await api.systemInfo()
    } catch {
      // non-fatal
    }
  }

  async function loadProjects() {
    loadingProjects.value = true
    error.value = null
    try {
      projects.value = await api.listProjects()
    } catch (err) {
      error.value = err.message
    } finally {
      loadingProjects.value = false
    }
  }

  async function loadRuns(project) {
    if (!project) return []
    const runs = await api.listRuns(project, { sort: "-updated_at_ms" })
    runsByProject.value = { ...runsByProject.value, [project]: runs }
    // Eagerly fetch batch metadata for metric/media/table/dist discovery.
    const missing = runs
      .map((r) => r.run_id)
      .filter((rid) => !detailsByKey.value[`${project}/${rid}`])
    if (missing.length > 0) {
      const batch = await api.batchRunMeta(project, missing.slice(0, 40))
      const next = { ...detailsByKey.value }
      for (const [rid, detail] of Object.entries(batch)) {
        next[`${project}/${rid}`] = detail
      }
      detailsByKey.value = next
    }
    ensureSse(project)
    return runs
  }

  async function loadRunDetail(project, runId, { force = false } = {}) {
    const key = `${project}/${runId}`
    if (!force && detailsByKey.value[key]) return detailsByKey.value[key]
    const detail = await api.getRun(project, runId)
    detailsByKey.value = { ...detailsByKey.value, [key]: detail }
    return detail
  }

  async function patchRun(project, runId, patch) {
    const updated = await api.patchRun(project, runId, patch)
    await loadRuns(project)
    await loadRunDetail(project, runId, { force: true })
    return updated
  }

  async function deleteRun(project, runId) {
    await api.deleteRun(project, runId)
    const next = { ...detailsByKey.value }
    delete next[`${project}/${runId}`]
    detailsByKey.value = next
    await loadProjects()
    await loadRuns(project)
  }

  function ensureSse(project) {
    if (typeof EventSource === "undefined" || _sseByProject.has(project)) return
    const es = api.subscribeProjectEvents(project, async (msg) => {
      sseConnected.value = true
      if (msg.changed && msg.changed.length > 0) {
        await loadRuns(project)
        for (const cb of _listeners) cb(project, msg)
      }
    })
    es.onopen = () => {
      sseConnected.value = true
    }
    _sseByProject.set(project, es)
  }

  function getRuns(project) {
    return runsByProject.value[project] || []
  }

  function getDetail(project, runId) {
    return detailsByKey.value[`${project}/${runId}`] || null
  }

  return {
    systemInfo,
    projects,
    runsByProject,
    detailsByKey,
    loadingProjects,
    error,
    sseConnected,
    onRunUpdates,
    loadSystem,
    loadProjects,
    loadRuns,
    loadRunDetail,
    patchRun,
    deleteRun,
    getRuns,
    getDetail,
  }
})
