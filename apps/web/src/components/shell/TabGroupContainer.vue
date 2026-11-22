<script setup>
import { ref } from "vue"
import { useTabsStore } from "@/stores/tabs"
import TabGroup from "./TabGroup.vue"

const props = defineProps({
  node: { type: Object, required: true },
})

const tabs = useTabsStore()
const containerEl = ref(null)

function startResize(ev) {
  if (!containerEl.value) return
  ev.preventDefault()
  const rect = containerEl.value.getBoundingClientRect()
  const isH = props.node.dir === "h"
  const onMove = (e) => {
    const pos = isH ? e.clientX - rect.left : e.clientY - rect.top
    const total = isH ? rect.width : rect.height
    if (total > 0) tabs.resizeSplit(props.node.id, (pos / total) * 100)
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
  <div
    v-if="node.kind === 'split'"
    ref="containerEl"
    class="w-full h-full flex min-w-0 min-h-0 overflow-hidden"
    :class="node.dir === 'h' ? 'flex-row' : 'flex-col'"
  >
    <div
      class="min-w-0 min-h-0 overflow-hidden"
      :style="node.dir === 'h' ? { width: `${node.ratio}%` } : { height: `${node.ratio}%` }"
    >
      <TabGroupContainer :node="node.a" />
    </div>
    <div
      :class="node.dir === 'h' ? 'kt-splitter-h' : 'kt-splitter-v'"
      @mousedown="startResize"
    />
    <div class="flex-1 min-w-0 min-h-0 overflow-hidden">
      <TabGroupContainer :node="node.b" />
    </div>
  </div>

  <TabGroup v-else :group-id="node.groupId" />
</template>
