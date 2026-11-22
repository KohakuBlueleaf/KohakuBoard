import { defineStore } from "pinia"
import { ref } from "vue"
import { colorForRun } from "@/utils/palette"

const COLORS_STORAGE_KEY = "kobo.runColors.v2"

function loadCustomColors() {
  if (typeof localStorage === "undefined") return {}
  try {
    const raw = localStorage.getItem(COLORS_STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function createDefaultScopeState() {
  return {
    hiddenRuns: {},
    runSearch: "",
    stateFilter: "all",
    groupBy: "none",
    legendMode: "name", // "name" | "run_id"
    diffOnlyConfig: false,
  }
}

export const useWorkspaceStore = defineStore("workspace", () => {
  const scopes = ref({})
  const customColors = ref(loadCustomColors())

  function getScope(scopeId) {
    if (!scopes.value[scopeId]) {
      scopes.value[scopeId] = createDefaultScopeState()
    }
    return scopes.value[scopeId]
  }

  function toggleRunVisibility(scopeId, runId) {
    const s = getScope(scopeId)
    s.hiddenRuns[runId] = !s.hiddenRuns[runId]
  }

  function setAllRunsVisibility(scopeId, runIds, visible) {
    const s = getScope(scopeId)
    for (const rid of runIds) {
      s.hiddenRuns[rid] = !visible
    }
  }

  function isRunVisible(scopeId, runId) {
    return !getScope(scopeId).hiddenRuns[runId]
  }

  function filteredRuns(scopeId, allRuns, lockedRunId = null) {
    if (lockedRunId) {
      return allRuns.filter((r) => r.run_id === lockedRunId)
    }
    const s = getScope(scopeId)
    const q = s.runSearch.trim().toLowerCase()
    return allRuns.filter((r) => {
      if (s.stateFilter !== "all" && r.state !== s.stateFilter) return false
      if (!q) return true
      return (
        r.name.toLowerCase().includes(q) ||
        r.run_id.toLowerCase().includes(q) ||
        (r.group && r.group.toLowerCase().includes(q)) ||
        (r.tags || []).some((t) => t.toLowerCase().includes(q))
      )
    })
  }

  function visibleRuns(scopeId, allRuns, lockedRunId = null) {
    const list = filteredRuns(scopeId, allRuns, lockedRunId)
    if (lockedRunId) return list
    return list.filter((r) => isRunVisible(scopeId, r.run_id))
  }

  function runColor(runId, allRuns) {
    if (customColors.value[runId]) return customColors.value[runId]
    const idx = allRuns.findIndex((r) => r.run_id === runId)
    return colorForRun(runId, idx)
  }

  function setRunColor(runId, color) {
    customColors.value[runId] = color
    if (typeof localStorage !== "undefined") {
      try {
        localStorage.setItem(COLORS_STORAGE_KEY, JSON.stringify(customColors.value))
      } catch {
        // ignore storage quota errors
      }
    }
  }

  function runLabel(scopeId, run) {
    const s = getScope(scopeId)
    if (s.legendMode === "run_id") return run.run_id
    return run.name || run.run_id
  }

  return {
    scopes,
    customColors,
    getScope,
    toggleRunVisibility,
    setAllRunsVisibility,
    isRunVisible,
    filteredRuns,
    visibleRuns,
    runColor,
    setRunColor,
    runLabel,
  }
})
