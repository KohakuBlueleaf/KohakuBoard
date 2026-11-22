<script setup>
import { computed } from "vue"
import { useTabsStore } from "@/stores/tabs"

const props = defineProps({
  groupId: { type: String, required: true },
})

const tabs = useTabsStore()

const group = computed(() => tabs.groups[props.groupId])
const stripTabs = computed(() => group.value?.tabs || [])
const activeId = computed(() => group.value?.activeId || null)

function onDragStart(tab, ev) {
  tabs.draggingTab = { tabId: tab.id, fromGroupId: props.groupId }
  if (ev.dataTransfer) {
    ev.dataTransfer.effectAllowed = "move"
    ev.dataTransfer.setData("application/x-kobo-tab", tab.id)
  }
}

function onDragEnd() {
  tabs.draggingTab = null
}

function onDropOnTab(targetTab) {
  const drag = tabs.draggingTab
  if (!drag) return
  if (drag.fromGroupId === props.groupId) {
    tabs.reorderTab(props.groupId, drag.tabId, targetTab.id)
  } else {
    const idx = stripTabs.value.findIndex((t) => t.id === targetTab.id)
    tabs.moveTabToGroup(drag.tabId, props.groupId, idx)
  }
  tabs.draggingTab = null
}

function onDropOnStrip() {
  const drag = tabs.draggingTab
  if (!drag) return
  tabs.moveTabToGroup(drag.tabId, props.groupId, -1)
  tabs.draggingTab = null
}
</script>

<template>
  <header
    class="h-9 kt-v2-chrome px-2 pt-1.5 flex items-end gap-0.5 overflow-x-auto shrink-0 select-none border-b kt-v2-line"
    data-testid="tab-strip"
    @dragover.prevent
    @drop.prevent="onDropOnStrip"
  >
    <div
      v-for="tab in stripTabs"
      :key="tab.id"
      draggable="true"
      class="kt-v2-tab px-3 py-1 flex items-center gap-2 text-xs max-w-[210px] cursor-pointer"
      :class="
        activeId === tab.id
          ? 'kt-v2-tab--active text-[var(--v2-ink)] font-semibold'
          : 'text-[var(--v2-ink-2)]'
      "
      title="Drag onto another tab to reorder, or drag onto pane edge to split"
      @click="tabs.activateTab(tab.id, groupId)"
      @dragstart="onDragStart(tab, $event)"
      @dragend="onDragEnd"
      @dragover.prevent
      @drop.stop.prevent="onDropOnTab(tab)"
    >
      <span
        class="w-1.5 h-1.5 rounded-full shrink-0"
        :class="tab.kind === 'run' ? 'bg-aquamarine' : 'bg-iolite'"
      />
      <span class="truncate">{{ tab.title }}</span>
      <button
        class="text-[10px] opacity-55 hover:opacity-100 px-0.5 rounded cursor-pointer"
        title="Close tab"
        @click.stop="tabs.closeTab(tab.id)"
      >
        <span class="i-carbon-close text-xs" />
      </button>
    </div>
  </header>
</template>
