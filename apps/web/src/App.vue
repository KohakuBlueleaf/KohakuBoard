<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from "vue"
import { useProjectsStore } from "@/stores/projects"
import { useSeriesCacheStore } from "@/stores/seriesCache"
import { useTabsStore } from "@/stores/tabs"
import { useThemeStore } from "@/stores/theme"
import { initializeSliderSync } from "@/composables/useSliderSync"
import RailPane from "@/components/shell/RailPane.vue"
import TabGroupContainer from "@/components/shell/TabGroupContainer.vue"
import CommandPalette from "@/components/chrome/CommandPalette.vue"

useThemeStore()
const projects = useProjectsStore()
const tabs = useTabsStore()
const seriesCache = useSeriesCacheStore()

const paletteOpen = ref(false)
let unsubSse = null

function onKeydown(ev) {
  if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "k") {
    ev.preventDefault()
    paletteOpen.value = !paletteOpen.value
  }
}

/**
 * Parse URL hash routes:
 *   #/p/<project>
 *   #/p/<project>/r/<runId>
 */
function parseHashRoute() {
  const hash = (window.location.hash || "").replace(/^#/, "")
  const mRun = hash.match(/^\/p\/([^/]+)\/r\/([^/]+)$/)
  if (mRun) {
    return { kind: "run", project: decodeURIComponent(mRun[1]), runId: decodeURIComponent(mRun[2]) }
  }
  const mProj = hash.match(/^\/p\/([^/]+)$/)
  if (mProj) {
    return { kind: "project", project: decodeURIComponent(mProj[1]) }
  }
  return null
}

function syncHashFromActiveTab(tab) {
  if (!tab) return
  const target =
    tab.kind === "run"
      ? `#/p/${encodeURIComponent(tab.project)}/r/${encodeURIComponent(tab.runId)}`
      : `#/p/${encodeURIComponent(tab.project)}`
  if (window.location.hash !== target) {
    window.history.replaceState(null, "", target)
  }
}

async function applyHashRoute() {
  const parsed = parseHashRoute()
  if (!parsed) return false
  await projects.loadRuns(parsed.project)
  if (parsed.kind === "run") {
    const run = projects.getRuns(parsed.project).find((r) => r.run_id === parsed.runId)
    tabs.openRunTab(parsed.project, parsed.runId, run?.name || parsed.runId)
    await projects.loadRunDetail(parsed.project, parsed.runId)
    return true
  }
  tabs.openProjectTab(parsed.project)
  return true
}

watch(
  () => tabs.activeTab,
  (tab) => {
    if (tab) syncHashFromActiveTab(tab)
  },
)

onMounted(async () => {
  initializeSliderSync()
  window.addEventListener("keydown", onKeydown)
  window.addEventListener("hashchange", applyHashRoute)
  unsubSse = projects.onRunUpdates((proj, msg) => {
    for (const item of msg.changed || []) {
      seriesCache.pollLiveTail(proj, item.run_id)
    }
  })

  await Promise.all([projects.loadSystem(), projects.loadProjects()])
  const restored = await applyHashRoute()
  if (!restored && projects.projects.length > 0 && tabs.allTabs.length === 0) {
    const first = projects.projects[0].name
    tabs.openProjectTab(first)
    await projects.loadRuns(first)
  }
})

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKeydown)
  window.removeEventListener("hashchange", applyHashRoute)
  if (unsubSse) unsubSse()
})
</script>

<template>
  <div class="w-full h-full flex overflow-hidden kt-v2-chrome">
    <RailPane @open-palette="paletteOpen = true" />

    <main class="flex-1 min-w-0 min-h-0 overflow-hidden">
      <TabGroupContainer :node="tabs.tree" />
    </main>

    <CommandPalette :open="paletteOpen" @close="paletteOpen = false" />
  </div>
</template>
