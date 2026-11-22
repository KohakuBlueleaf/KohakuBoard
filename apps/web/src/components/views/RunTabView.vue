<script setup>
import { computed, nextTick, ref, watch } from "vue"
import { useProjectsStore } from "@/stores/projects"
import { useTabsStore } from "@/stores/tabs"
import { statusColor } from "@/utils/palette"
import { formatBytes, formatCount, formatDuration } from "@/utils/format"
import MultimodalBoard from "@/components/board/MultimodalBoard.vue"
import RunOverviewPanel from "@/components/panels/RunOverviewPanel.vue"

const props = defineProps({
  tab: { type: Object, required: true },
})

const projects = useProjectsStore()
const tabs = useTabsStore()

const inspectorOpen = ref(false)
const inspectorWidth = ref(340)

// Inline run name editing
const isEditingName = ref(false)
const nameDraft = ref("")
const nameInputEl = ref(null)
const savingName = ref(false)

const allRuns = computed(() => projects.getRuns(props.tab.project))
const runSummary = computed(() => allRuns.value.find((r) => r.run_id === props.tab.runId) || null)
const detail = computed(() => projects.getDetail(props.tab.project, props.tab.runId))

const displayName = computed(
  () => detail.value?.info?.name || runSummary.value?.name || props.tab.title || props.tab.runId,
)
const runState = computed(() => detail.value?.info?.state || runSummary.value?.state || "finished")

watch(
  () => [props.tab.project, props.tab.runId],
  async ([proj, rid]) => {
    if (!proj || !rid) return
    if (projects.getRuns(proj).length === 0) {
      await projects.loadRuns(proj)
    }
    await projects.loadRunDetail(proj, rid)
  },
  { immediate: true },
)

function startEditName() {
  nameDraft.value = displayName.value
  isEditingName.value = true
  nextTick(() => {
    nameInputEl.value?.focus()
    nameInputEl.value?.select()
  })
}

async function saveName() {
  if (!isEditingName.value || savingName.value) return
  const clean = nameDraft.value.trim()
  isEditingName.value = false
  if (!clean || clean === displayName.value) return
  savingName.value = true
  try {
    await projects.patchRun(props.tab.project, props.tab.runId, { name: clean })
    tabs.openRunTab(props.tab.project, props.tab.runId, clean)
  } finally {
    savingName.value = false
  }
}

function startInspectorResize(ev) {
  ev.preventDefault()
  const startX = ev.clientX
  const startW = inspectorWidth.value
  const onMove = (e) => {
    inspectorWidth.value = Math.max(240, Math.min(540, Math.round(startW - (e.clientX - startX))))
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
  <div class="w-full h-full flex flex-col overflow-hidden" data-testid="run-tab-view">
    <!-- Dedicated First-Class Run Header -->
    <div class="px-3 py-2 border-b kt-v2-edge kt-v2-panel flex flex-wrap items-center justify-between gap-3 shrink-0">
      <div class="flex items-center gap-2.5 min-w-0">
        <button
          class="text-xs text-[var(--v2-ink-2)] hover:text-iolite flex items-center gap-1 cursor-pointer shrink-0"
          title="Open Project Comparison View"
          @click="tabs.openProjectTab(tab.project)"
        >
          <span class="i-carbon-folder text-xs" />
          <span>{{ tab.project }}</span>
        </button>
        <span class="text-[var(--v2-ink-3)]">/</span>

        <input
          v-if="isEditingName"
          ref="nameInputEl"
          v-model="nameDraft"
          type="text"
          class="px-2 py-0.5 text-sm font-semibold rounded border border-iolite bg-[var(--v2-card)] text-[var(--v2-ink)] outline-none"
          @keydown.enter="saveName"
          @keydown.esc="isEditingName = false"
          @blur="saveName"
        />
        <span
          v-else
          class="text-sm font-semibold truncate cursor-pointer hover:text-iolite"
          title="Double-click to rename run"
          data-testid="run-header-title"
          @dblclick="startEditName"
        >
          {{ displayName }}
        </span>

        <span
          class="px-2 py-0.5 rounded-full text-[10px] font-medium text-white shrink-0"
          :style="{ background: statusColor(runState) }"
        >
          {{ runState }}
        </span>

        <span class="num text-[11px] text-[var(--v2-ink-3)] hidden md:inline">
          {{ tab.runId }}
          <template v-if="runSummary"> · {{ formatCount(runSummary.last_step + 1) }} steps</template>
          <template v-if="detail?.info">
            · {{ formatDuration(detail.info.created_at_ms, detail.info.finished_at_ms) }}
          </template>
          <template v-if="detail?.storage">
            · {{ formatBytes((detail.storage.run_db_bytes || 0) + (detail.storage.blobs_db_bytes || 0)) }}
          </template>
        </span>
      </div>

      <div class="flex items-center gap-1.5 shrink-0">
        <button
          class="btn-secondary text-[11px] py-1 px-2 flex items-center gap-1"
          data-testid="toggle-dossier-btn"
          @click="inspectorOpen = !inspectorOpen"
        >
          <span class="i-carbon-information text-xs" />
          {{ inspectorOpen ? "Hide Dossier" : "Run Dossier" }}
        </button>
      </div>
    </div>

    <!-- Main Per-Run Multimodal Board + Collapsible Right Dossier -->
    <div class="flex-1 min-h-0 flex overflow-hidden">
      <div class="flex-1 min-w-0 min-h-0 overflow-hidden">
        <MultimodalBoard
          mode="run"
          :scope-id="tab.id"
          :project="tab.project"
          :locked-run-id="tab.runId"
        />
      </div>

      <template v-if="inspectorOpen">
        <div class="kt-splitter-h" @mousedown="startInspectorResize" />
        <div class="h-full shrink-0 overflow-hidden" :style="{ width: `${inspectorWidth}px` }">
          <RunOverviewPanel
            :scope-id="tab.id"
            :project="tab.project"
            :locked-run-id="tab.runId"
          />
        </div>
      </template>
    </div>
  </div>
</template>
