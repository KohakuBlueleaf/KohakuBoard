<script setup>
import { computed, ref } from "vue"
import { useTabsStore } from "@/stores/tabs"
import TabStrip from "./TabStrip.vue"
import ProjectTabView from "@/components/views/ProjectTabView.vue"
import RunTabView from "@/components/views/RunTabView.vue"

const props = defineProps({
  groupId: { type: String, required: true },
})

const tabs = useTabsStore()
const bodyEl = ref(null)
const hoverEdge = ref(null)

const group = computed(() => tabs.groups[props.groupId])
const activeTab = computed(() => {
  const g = group.value
  if (!g) return null
  return g.tabs.find((t) => t.id === g.activeId) || g.tabs[0] || null
})
const focused = computed(() => tabs.focusedGroupId === props.groupId && tabs.groupCount > 1)

function detectEdge(ev) {
  if (!bodyEl.value) return "center"
  const rect = bodyEl.value.getBoundingClientRect()
  const rx = (ev.clientX - rect.left) / Math.max(1, rect.width)
  const ry = (ev.clientY - rect.top) / Math.max(1, rect.height)
  if (rx < 0.25) return "left"
  if (rx > 0.75) return "right"
  if (ry < 0.25) return "top"
  if (ry > 0.75) return "bottom"
  return "center"
}

function onBodyDragOver(ev) {
  if (!tabs.draggingTab) return
  ev.preventDefault()
  hoverEdge.value = detectEdge(ev)
}

function onBodyDragLeave() {
  hoverEdge.value = null
}

function onBodyDrop(ev) {
  if (!tabs.draggingTab) return
  ev.preventDefault()
  const edge = hoverEdge.value || detectEdge(ev)
  tabs.splitTabToEdge(tabs.draggingTab.tabId, props.groupId, edge)
  tabs.draggingTab = null
  hoverEdge.value = null
}

const overlayClass = computed(() => {
  switch (hoverEdge.value) {
    case "left":
      return "inset-y-0 left-0 w-1/2"
    case "right":
      return "inset-y-0 right-0 w-1/2"
    case "top":
      return "inset-x-0 top-0 h-1/2"
    case "bottom":
      return "inset-x-0 bottom-0 h-1/2"
    case "center":
      return "inset-0"
    default:
      return "hidden"
  }
})
</script>

<template>
  <div
    class="relative h-full flex flex-col overflow-hidden kt-v2-canvas"
    :class="focused ? 'outline outline-1 -outline-offset-1 outline-iolite/45' : ''"
    @pointerdown="tabs.setFocusedGroup(groupId)"
  >
    <TabStrip :group-id="groupId" />

    <div
      ref="bodyEl"
      class="relative flex-1 min-h-0 min-w-0 overflow-hidden"
      @dragover="onBodyDragOver"
      @dragleave="onBodyDragLeave"
      @drop="onBodyDrop"
    >
      <ProjectTabView
        v-if="activeTab && activeTab.kind === 'project'"
        :key="activeTab.id"
        :tab="activeTab"
      />
      <RunTabView
        v-else-if="activeTab && activeTab.kind === 'run'"
        :key="activeTab.id"
        :tab="activeTab"
      />
      <div
        v-else
        class="w-full h-full flex flex-col items-center justify-center gap-3 text-center p-6"
      >
        <div
          class="w-12 h-12 rounded-2xl bg-iolite text-white flex items-center justify-center text-xl font-bold"
        >
          K
        </div>
        <div class="text-base font-semibold">
          <span class="font-bold text-amber">Kohaku</span>
          <span class="font-light text-iolite dark:text-iolite-light">Board</span>
        </div>
        <div class="text-xs text-[var(--v2-ink-2)] max-w-sm">
          Select a project from the left rail or start logging a run with
          <code class="px-1 py-0.5 rounded bg-[var(--v2-panel)]">kobo.init()</code>.
        </div>
      </div>

      <!-- Edge-drop split overlay (KT v2 TabGroup drag-to-split) -->
      <div
        v-if="hoverEdge"
        class="pointer-events-none absolute z-30 bg-iolite/20 border-2 border-iolite/60 rounded transition-all"
        :class="overlayClass"
      />
    </div>
  </div>
</template>
