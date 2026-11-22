<script setup>
import { ref } from "vue"
import RunsSidebarPanel from "@/components/panels/RunsSidebarPanel.vue"
import MultimodalBoard from "@/components/board/MultimodalBoard.vue"

defineProps({
  tab: { type: Object, required: true },
})

const sidebarOpen = ref(true)
const sidebarWidth = ref(248)

function startSidebarResize(ev) {
  ev.preventDefault()
  const startX = ev.clientX
  const startW = sidebarWidth.value
  const onMove = (e) => {
    sidebarWidth.value = Math.max(180, Math.min(440, Math.round(startW + (e.clientX - startX))))
  }
  const onUp = () => {
    window.removeEventListener("mousemove", onMove)
    window.removeEventListener("mouseup", onUp)
  }
  window.addEventListener("mousemove", onMove)
  window.addEventListener("mouseup", onUp)
}
</script>

<template>
  <div class="w-full h-full flex overflow-hidden" data-testid="project-tab-view">
    <!-- Resizable Project Runs Sidebar -->
    <template v-if="sidebarOpen">
      <div class="h-full shrink-0 overflow-hidden" :style="{ width: `${sidebarWidth}px` }">
        <RunsSidebarPanel :scope-id="tab.id" :project="tab.project" />
      </div>
      <div class="kt-splitter-h" @mousedown="startSidebarResize" />
    </template>

    <!-- Unified Multimodal Board (auto-namespace tabs + collapsible sections + built-in Runs Table tab) -->
    <div class="flex-1 min-w-0 min-h-0 overflow-hidden">
      <MultimodalBoard mode="project" :scope-id="tab.id" :project="tab.project">
        <template #strip-prefix>
          <button
            class="btn-ghost px-1.5 py-1 mb-0.5 mr-1 shrink-0"
            :title="sidebarOpen ? 'Hide runs sidebar' : 'Show runs sidebar'"
            @click="sidebarOpen = !sidebarOpen"
          >
            <span :class="sidebarOpen ? 'i-carbon-side-panel-close' : 'i-carbon-side-panel-open'" />
          </button>
        </template>
      </MultimodalBoard>
    </div>
  </div>
</template>
