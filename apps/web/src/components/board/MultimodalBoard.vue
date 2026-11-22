<script setup>
import { computed, nextTick, ref, watch } from "vue"
import { useBoardLayoutStore } from "@/stores/boardLayout"
import { useProjectsStore } from "@/stores/projects"
import { useSeriesCacheStore } from "@/stores/seriesCache"
import { useWorkspaceStore } from "@/stores/workspace"
import BoardCard from "./BoardCard.vue"
import RunsTablePanel from "@/components/panels/RunsTablePanel.vue"
import ConsoleLogsPanel from "@/components/panels/ConsoleLogsPanel.vue"

const props = defineProps({
  mode: { type: String, required: true }, // "project" | "run"
  scopeId: { type: String, required: true },
  project: { type: String, required: true },
  lockedRunId: { type: String, default: null },
})

const projects = useProjectsStore()
const workspace = useWorkspaceStore()
const boardStore = useBoardLayoutStore()
const seriesCache = useSeriesCacheStore()

const searchQuery = ref("")
const hoverSync = ref(true)
const dropTargetTabId = ref(null)
const dropTargetSectionId = ref(null)

// Inline Tab & Section rename state
const editingTabId = ref(null)
const editingTabName = ref("")
const editingSectionId = ref(null)
const editingSectionTitle = ref("")

// Modals: + Tab, + Section, + Panel (multimodal!)
const addTabOpen = ref(false)
const newTabName = ref("")

const addSectionOpen = ref(false)
const newSectionTitle = ref("")

const addPanelOpen = ref(false)
const newPanelTargetSectionId = ref("")
const newPanelKind = ref("scalar") // "scalar" | "media" | "distribution" | "table"
const newPanelTitle = ref("")
const newPanelMetrics = ref([])
const newPanelSingleKey = ref("")

// Layout scope key: per-project OR per-run!
const boardScopeKey = computed(() =>
  props.mode === "run" && props.lockedRunId
    ? `run:${props.project}:${props.lockedRunId}`
    : `proj:${props.project}`,
)
const templateKey = computed(() => (props.mode === "run" ? `run-tpl:${props.project}` : null))

const allRuns = computed(() => projects.getRuns(props.project))
const visibleRuns = computed(() =>
  workspace.visibleRuns(props.scopeId, allRuns.value, props.lockedRunId).map((r) => ({
    run_id: r.run_id,
    name: r.name,
    label: workspace.runLabel(props.scopeId, r),
    color: workspace.runColor(r.run_id, allRuns.value),
  })),
)

// Aggregate ALL modalities across visible runs!
const catalog = computed(() => {
  const scalars = new Set()
  const media = new Set()
  const histograms = new Set()
  const kdes = new Set()
  const tables = new Set()

  for (const r of visibleRuns.value) {
    const detail = projects.getDetail(props.project, r.run_id)
    if (!detail) continue
    for (const m of detail.scalar_names || []) scalars.add(m)
    for (const m of Object.keys(detail.summary || {})) scalars.add(m)
    for (const m of detail.media_names || []) media.add(m)
    for (const h of detail.histogram_names || []) histograms.add(h)
    for (const k of detail.kde_names || []) kdes.add(k)
    for (const t of detail.table_names || []) tables.add(t)
  }

  return {
    scalars: Array.from(scalars).sort(),
    media: Array.from(media).sort(),
    histograms: Array.from(histograms).sort(),
    kdes: Array.from(kdes).sort(),
    tables: Array.from(tables).sort(),
  }
})

const distributionOptions = computed(() => [
  ...catalog.value.histograms.map((h) => ({ key: `histogram:${h}`, kind: "histogram", name: h })),
  ...catalog.value.kdes.map((k) => ({ key: `kde:${k}`, kind: "kde", name: k })),
])

watch(
  [boardScopeKey, catalog],
  ([key, cat]) => {
    boardStore.syncCatalog(key, cat, templateKey.value)
  },
  { immediate: true, deep: true },
)

const boardState = computed(() => boardStore.ensure(boardScopeKey.value, templateKey.value))

// Built-in tab: "Runs Table" for project view, "Logs" for run view!
const builtinTab = computed(() =>
  props.mode === "project"
    ? { id: "builtin:runs-table", name: "Runs Table", icon: "i-carbon-table-split" }
    : { id: "builtin:logs", name: "Logs", icon: "i-carbon-terminal" },
)

const isBuiltinActive = computed(() => boardState.value.activeTabId === builtinTab.value.id)

const activeBoardTab = computed(() => {
  if (isBuiltinActive.value) return null
  return (
    boardState.value.tabs.find((t) => t.id === boardState.value.activeTabId) ||
    boardState.value.tabs[0] ||
    null
  )
})

const QUICK_WIDTHS = [
  { p: 25.0, l: "1/4" },
  { p: 33.333, l: "1/3" },
  { p: 50.0, l: "1/2" },
  { p: 100.0, l: "Full" },
]

function filteredSectionCards(section) {
  const q = searchQuery.value.trim().toLowerCase()
  if (!q) return section.cards
  return section.cards.filter((c) => {
    if ((c.title || "").toLowerCase().includes(q)) return true
    if (c.kind === "scalar" && (c.metrics || []).some((m) => m.toLowerCase().includes(q))) return true
    if (c.mediaName && c.mediaName.toLowerCase().includes(q)) return true
    if (c.distName && c.distName.toLowerCase().includes(q)) return true
    if (c.tableName && c.tableName.toLowerCase().includes(q)) return true
    return false
  })
}

// ---- Drag Card onto Tab Pill (moves card into that Tab!) ----

function onTabDragOver(tabId, ev) {
  const drag = boardStore.draggingCard
  if (!drag || drag.scopeKey !== boardScopeKey.value) return
  if (drag.tabId === tabId) return
  ev.preventDefault()
  dropTargetTabId.value = tabId
}

function onTabDragLeave(tabId) {
  if (dropTargetTabId.value === tabId) dropTargetTabId.value = null
}

function onTabDrop(tabId, ev) {
  dropTargetTabId.value = null
  const drag = boardStore.draggingCard
  if (!drag || drag.scopeKey !== boardScopeKey.value) return
  ev.preventDefault()
  boardStore.moveCard(boardScopeKey.value, drag.tabId, drag.sectionId, drag.cardId, tabId, null, null)
  boardStore.draggingCard = null
}

// ---- Drag Card onto Section Header / Body ----

function onSectionDragOver(sectionId, ev) {
  const drag = boardStore.draggingCard
  if (!drag || drag.scopeKey !== boardScopeKey.value) return
  ev.preventDefault()
  dropTargetSectionId.value = sectionId
}

function onSectionDragLeave(sectionId) {
  if (dropTargetSectionId.value === sectionId) dropTargetSectionId.value = null
}

function onSectionDrop(sectionId, ev) {
  dropTargetSectionId.value = null
  const drag = boardStore.draggingCard
  if (!drag || drag.scopeKey !== boardScopeKey.value || !activeBoardTab.value) return
  ev.preventDefault()
  boardStore.moveCard(
    boardScopeKey.value,
    drag.tabId,
    drag.sectionId,
    drag.cardId,
    activeBoardTab.value.id,
    sectionId,
    null,
  )
  boardStore.draggingCard = null
}

function onCardDropOnCard({ targetSectionId, targetCardId }) {
  const drag = boardStore.draggingCard
  if (!drag || drag.scopeKey !== boardScopeKey.value || !activeBoardTab.value) return
  boardStore.moveCard(
    boardScopeKey.value,
    drag.tabId,
    drag.sectionId,
    drag.cardId,
    activeBoardTab.value.id,
    targetSectionId,
    targetCardId,
  )
  boardStore.draggingCard = null
}

// ---- Inline Rename Tabs & Sections ----

function startRenameTab(tab) {
  editingTabId.value = tab.id
  editingTabName.value = tab.name
  nextTick(() => document.getElementById(`tab-rename-${tab.id}`)?.focus())
}

function commitRenameTab(tabId) {
  if (editingTabId.value !== tabId) return
  boardStore.renameTab(boardScopeKey.value, tabId, editingTabName.value)
  editingTabId.value = null
}

function startRenameSection(sec) {
  editingSectionId.value = sec.id
  editingSectionTitle.value = sec.title
  nextTick(() => document.getElementById(`sec-rename-${sec.id}`)?.focus())
}

function commitRenameSection(secId) {
  if (editingSectionId.value !== secId || !activeBoardTab.value) return
  boardStore.renameSection(boardScopeKey.value, activeBoardTab.value.id, secId, editingSectionTitle.value)
  editingSectionId.value = null
}

// ---- Add Tab / Add Section / Add Multimodal Panel ----

function openAddTab() {
  newTabName.value = ""
  addTabOpen.value = true
}

function confirmAddTab() {
  boardStore.addTab(boardScopeKey.value, newTabName.value)
  addTabOpen.value = false
}

function openAddSection() {
  newSectionTitle.value = ""
  addSectionOpen.value = true
}

function confirmAddSection() {
  if (!activeBoardTab.value) return
  boardStore.addSection(boardScopeKey.value, activeBoardTab.value.id, newSectionTitle.value)
  addSectionOpen.value = false
}

function openAddPanel(sectionId = "") {
  if (!activeBoardTab.value) return
  newPanelTargetSectionId.value = sectionId || activeBoardTab.value.sections[0]?.id || ""
  newPanelKind.value = "scalar"
  newPanelTitle.value = ""
  newPanelMetrics.value = catalog.value.scalars.slice(0, Math.min(2, catalog.value.scalars.length))
  newPanelSingleKey.value = ""
  addPanelOpen.value = true
}

function onPanelKindChange(kind) {
  newPanelKind.value = kind
  newPanelSingleKey.value = ""
  if (kind === "media" && catalog.value.media.length > 0) newPanelSingleKey.value = catalog.value.media[0]
  if (kind === "distribution" && distributionOptions.value.length > 0) {
    newPanelSingleKey.value = distributionOptions.value[0].key
  }
  if (kind === "table" && catalog.value.tables.length > 0) newPanelSingleKey.value = catalog.value.tables[0]
}

function toggleNewMetric(m) {
  const idx = newPanelMetrics.value.indexOf(m)
  if (idx >= 0) newPanelMetrics.value.splice(idx, 1)
  else newPanelMetrics.value.push(m)
}

function confirmAddPanel() {
  if (!activeBoardTab.value) return
  let spec = null
  if (newPanelKind.value === "scalar") {
    if (newPanelMetrics.value.length === 0) return
    spec = {
      kind: "scalar",
      title: newPanelTitle.value.trim() || newPanelMetrics.value.join(" + "),
      metrics: [...newPanelMetrics.value],
    }
  } else if (newPanelKind.value === "media") {
    if (!newPanelSingleKey.value) return
    spec = {
      kind: "media",
      title: newPanelTitle.value.trim() || newPanelSingleKey.value,
      mediaName: newPanelSingleKey.value,
      currentStep: null,
      autoAdvance: true,
      height: 300,
    }
  } else if (newPanelKind.value === "distribution") {
    if (!newPanelSingleKey.value) return
    const idx = newPanelSingleKey.value.indexOf(":")
    const distKind = newPanelSingleKey.value.slice(0, idx)
    const distName = newPanelSingleKey.value.slice(idx + 1)
    spec = {
      kind: "distribution",
      distKind,
      distName,
      title: newPanelTitle.value.trim() || distName,
      histogramMode: activeBoardTab.value.settings.histogramMode || "single",
      normalize: "per-step",
      currentStep: null,
      autoAdvance: true,
      height: 280,
    }
  } else if (newPanelKind.value === "table") {
    if (!newPanelSingleKey.value) return
    spec = {
      kind: "table",
      title: newPanelTitle.value.trim() || newPanelSingleKey.value,
      tableName: newPanelSingleKey.value,
      currentStep: null,
      autoAdvance: true,
      widthPercent: 50.0,
      height: 300,
    }
  }
  if (!spec) return
  boardStore.addCard(boardScopeKey.value, activeBoardTab.value.id, newPanelTargetSectionId.value, spec)
  addPanelOpen.value = false
}

function updateSmoothingVal(val) {
  if (!activeBoardTab.value) return
  boardStore.updateTabSettings(boardScopeKey.value, activeBoardTab.value.id, {
    smoothingValue: Number(val),
  })
}

const smoothingSliderMax = computed(() =>
  activeBoardTab.value?.settings?.smoothingMode === "ema" ? 0.95 : 50,
)
const smoothingSliderStep = computed(() =>
  activeBoardTab.value?.settings?.smoothingMode === "ema" ? 0.05 : 1,
)
</script>

<template>
  <div class="h-full w-full flex flex-col kt-v2-canvas overflow-hidden" data-testid="multimodal-board">
    <!-- Board Tab Strip (auto-created namespace tabs + custom tabs + drop-card-to-tab + built-in Runs Table / Logs tab) -->
    <div
      class="px-3 pt-1.5 border-b kt-v2-edge flex items-center gap-1 overflow-x-auto shrink-0 bg-[var(--v2-panel)]/55"
      data-testid="board-tab-strip"
    >
      <slot name="strip-prefix" />

      <div
        v-for="tab in boardState.tabs"
        :key="tab.id"
        class="group relative px-2.5 py-1.5 rounded-t-lg text-xs font-medium flex items-center gap-1.5 whitespace-nowrap cursor-pointer border border-b-0 transition-all select-none"
        :class="[
          boardState.activeTabId === tab.id
            ? 'bg-[var(--v2-canvas)] text-[var(--v2-ink)] border-[var(--v2-edge)] -mb-px z-10'
            : 'border-transparent text-[var(--v2-ink-2)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-tab-hover)]',
          dropTargetTabId === tab.id
            ? 'ring-2 ring-iolite bg-iolite/15 text-iolite dark:text-iolite-light scale-[1.03]'
            : '',
        ]"
        :data-testid="`board-tab-${tab.name}`"
        @click="boardStore.setActiveTab(boardScopeKey, tab.id)"
        @dblclick.stop="startRenameTab(tab)"
        @dragover="onTabDragOver(tab.id, $event)"
        @dragleave="onTabDragLeave(tab.id)"
        @drop="onTabDrop(tab.id, $event)"
      >
        <input
          v-if="editingTabId === tab.id"
          :id="`tab-rename-${tab.id}`"
          v-model="editingTabName"
          type="text"
          class="w-24 px-1 py-0 text-xs rounded border border-iolite bg-[var(--v2-card)] text-[var(--v2-ink)] outline-none"
          @keydown.enter="commitRenameTab(tab.id)"
          @keydown.esc="editingTabId = null"
          @blur="commitRenameTab(tab.id)"
          @click.stop
        />
        <span v-else>{{ tab.name }}</span>

        <span
          class="text-[10px] num px-1 rounded bg-[var(--v2-panel)] text-[var(--v2-ink-3)]"
          :class="dropTargetTabId === tab.id ? 'bg-iolite text-white' : ''"
        >
          {{ dropTargetTabId === tab.id ? "+ drop" : boardStore.countCardsInTab(tab) }}
        </span>

        <button
          v-if="boardState.tabs.length > 1"
          class="w-4 h-4 rounded flex items-center justify-center opacity-0 group-hover:opacity-100 hover:bg-[var(--v2-line)] text-[var(--v2-ink-3)] hover:text-[var(--v2-ink)] cursor-pointer"
          title="Delete tab"
          @click.stop="boardStore.removeTab(boardScopeKey, tab.id)"
        >
          <span class="i-carbon-close text-[10px]" />
        </button>
      </div>

      <!-- + Add Tab button -->
      <button
        class="px-2 py-1 mb-0.5 rounded-md text-xs text-[var(--v2-ink-2)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-tab-hover)] flex items-center gap-1 cursor-pointer shrink-0"
        title="Create new custom tab"
        data-testid="add-board-tab-btn"
        @click="openAddTab"
      >
        <span class="i-carbon-add text-xs" />
        <span class="hidden sm:inline">Tab</span>
      </button>

      <!-- Spacer + Built-in special tab (Runs Table in Project View / Logs in Run View) -->
      <div class="ml-auto flex items-center gap-1 pl-2">
        <button
          class="px-2.5 py-1.5 rounded-t-lg text-xs font-medium flex items-center gap-1.5 whitespace-nowrap cursor-pointer border border-b-0 transition-colors"
          :class="
            isBuiltinActive
              ? 'bg-[var(--v2-canvas)] text-iolite dark:text-iolite-light border-[var(--v2-edge)] -mb-px z-10'
              : 'border-transparent text-[var(--v2-ink-2)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-tab-hover)]'
          "
          :data-testid="`builtin-tab-${mode}`"
          @click="boardStore.setActiveTab(boardScopeKey, builtinTab.id)"
        >
          <span :class="builtinTab.icon" class="text-xs" />
          {{ builtinTab.name }}
        </button>
        <slot name="strip-suffix" />
      </div>
    </div>

    <!-- Built-in Special View: Runs Table (Project View) OR Console Logs (Run View) -->
    <div v-if="isBuiltinActive" class="flex-1 min-h-0 overflow-hidden">
      <RunsTablePanel v-if="mode === 'project'" :scope-id="scopeId" :project="project" />
      <ConsoleLogsPanel v-else :scope-id="scopeId" :project="project" :locked-run-id="lockedRunId" />
    </div>

    <!-- Active Multimodal Board Tab -->
    <template v-else-if="activeBoardTab">
      <!-- Compact Tab Global Controls Toolbar -->
      <div class="px-3 py-1.5 border-b kt-v2-line flex flex-wrap items-center gap-2 shrink-0 text-xs">
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Filter panels..."
          class="w-36 px-2 py-1 text-xs rounded-lg border kt-v2-line bg-[var(--v2-card)] text-[var(--v2-ink)] outline-none focus:border-iolite"
        />

        <!-- X-axis (built-ins + any logged scalar metric!) -->
        <div class="flex items-center gap-1">
          <span class="text-[var(--v2-ink-3)]">X:</span>
          <select
            :value="activeBoardTab.settings.xAxis"
            class="px-1.5 py-1 text-xs rounded-md border kt-v2-line bg-[var(--v2-card)] text-[var(--v2-ink)] max-w-36"
            @change="
              boardStore.updateTabSettings(boardScopeKey, activeBoardTab.id, {
                xAxis: $event.target.value,
              })
            "
          >
            <option value="global_step">global_step</option>
            <option value="step">step</option>
            <option value="relative_time">time (s)</option>
            <option v-for="m in catalog.scalars" :key="m" :value="m">{{ m }}</option>
          </select>
        </div>

        <!-- Smoothing Mode + Slider -->
        <div class="flex items-center gap-1">
          <select
            :value="activeBoardTab.settings.smoothingMode"
            class="px-1.5 py-1 text-xs rounded-md border kt-v2-line bg-[var(--v2-card)] text-[var(--v2-ink)]"
            title="Smoothing mode"
            @change="
              boardStore.updateTabSettings(boardScopeKey, activeBoardTab.id, {
                smoothingMode: $event.target.value,
                smoothingValue: $event.target.value === 'ema' ? 0.6 : $event.target.value === 'none' ? 0 : 5,
              })
            "
          >
            <option value="ema">EMA</option>
            <option value="ma">MA</option>
            <option value="gaussian">Gauss</option>
            <option value="none">Raw</option>
          </select>
          <template v-if="activeBoardTab.settings.smoothingMode !== 'none'">
            <input
              :value="activeBoardTab.settings.smoothingValue"
              type="range"
              min="0"
              :max="smoothingSliderMax"
              :step="smoothingSliderStep"
              class="w-16 accent-iolite"
              @input="updateSmoothingVal($event.target.value)"
            />
            <span class="num w-7 text-[11px] text-[var(--v2-ink-2)]">
              {{
                activeBoardTab.settings.smoothingMode === "ema"
                  ? Number(activeBoardTab.settings.smoothingValue).toFixed(2)
                  : Math.round(activeBoardTab.settings.smoothingValue)
              }}
            </span>
          </template>
        </div>

        <button
          class="px-2 py-1 rounded-md border transition-colors cursor-pointer"
          :class="
            activeBoardTab.settings.logY
              ? 'bg-sapphire text-white border-sapphire'
              : 'kt-v2-line bg-[var(--v2-card)] text-[var(--v2-ink-2)]'
          "
          @click="
            boardStore.updateTabSettings(boardScopeKey, activeBoardTab.id, {
              logY: !activeBoardTab.settings.logY,
            })
          "
        >
          Log Y
        </button>

        <button
          class="px-2 py-1 rounded-md border transition-colors cursor-pointer"
          :class="
            hoverSync
              ? 'bg-aquamarine text-white border-aquamarine'
              : 'kt-v2-line bg-[var(--v2-card)] text-[var(--v2-ink-2)]'
          "
          title="Sync hover crosshair across charts"
          @click="hoverSync = !hoverSync"
        >
          Sync
        </button>

        <div class="flex items-center gap-1 ml-auto flex-wrap">
          <span
            v-if="seriesCache.streamingCount > 0"
            class="text-[10px] px-2 py-0.5 rounded-full bg-aquamarine-light text-aquamarine-shadow dark:bg-aquamarine-shadow/40 dark:text-aquamarine-light num"
          >
            streaming…
          </span>

          <div class="hidden md:flex items-center gap-0.5 mr-1">
            <button
              v-for="w in QUICK_WIDTHS"
              :key="w.l"
              class="px-1.5 py-0.5 rounded text-[11px] num cursor-pointer border"
              :class="
                Math.abs(boardState.defaultWidth - w.p) < 0.8
                  ? 'bg-iolite text-white border-iolite'
                  : 'bg-[var(--v2-card)] text-[var(--v2-ink-2)] kt-v2-line'
              "
              title="Set width for all panels in this tab"
              @click="boardStore.setTabCardsWidth(boardScopeKey, activeBoardTab.id, w.p)"
            >
              {{ w.l }}
            </button>
          </div>

          <button
            class="btn-secondary flex items-center gap-1 py-1 px-2 text-xs"
            data-testid="add-section-btn"
            title="Add collapsible section to this tab"
            @click="openAddSection"
          >
            <span class="i-carbon-row-insert text-xs" />
            Section
          </button>

          <button
            class="btn-primary flex items-center gap-1 py-1 px-2 text-xs"
            data-testid="add-panel-btn"
            title="Add Scalar, Media, Distribution, or Table panel"
            @click="openAddPanel()"
          >
            <span class="i-carbon-add text-sm" />
            Panel
          </button>

          <button
            v-if="mode === 'run'"
            class="btn-ghost text-[11px]"
            title="Save this run's tabs & sections as default template for new runs in this project"
            @click="boardStore.saveAsTemplate(templateKey, boardScopeKey)"
          >
            Save Tpl
          </button>

          <button
            class="btn-ghost text-[11px]"
            title="Reset tabs, sections, and cards to auto-grouped defaults"
            @click="boardStore.resetBoard(boardScopeKey, catalog)"
          >
            Reset
          </button>
        </div>
      </div>

      <!-- Collapsible Sections Canvas -->
      <div class="flex-1 overflow-y-auto p-3 space-y-3" data-testid="sections-canvas">
        <section
          v-for="sec in activeBoardTab.sections"
          :key="sec.id"
          class="rounded-xl border transition-colors"
          :class="
            dropTargetSectionId === sec.id
              ? 'border-iolite bg-iolite/6'
              : 'border-[var(--v2-line)] bg-[var(--v2-panel)]/25'
          "
          :data-testid="`board-section-${sec.title}`"
          @dragover="onSectionDragOver(sec.id, $event)"
          @dragleave="onSectionDragLeave(sec.id)"
          @drop="onSectionDrop(sec.id, $event)"
        >
          <!-- W&B-style Collapsible Section Header -->
          <div
            class="px-3 py-1.5 flex items-center justify-between gap-2 cursor-pointer select-none border-b"
            :class="sec.collapsed ? 'border-transparent' : 'kt-v2-line/80'"
            @click="boardStore.toggleSection(boardScopeKey, activeBoardTab.id, sec.id)"
          >
            <div class="flex items-center gap-1.5 min-w-0">
              <span
                :class="sec.collapsed ? 'i-carbon-chevron-right' : 'i-carbon-chevron-down'"
                class="text-xs text-[var(--v2-ink-2)] shrink-0"
              />
              <input
                v-if="editingSectionId === sec.id"
                :id="`sec-rename-${sec.id}`"
                v-model="editingSectionTitle"
                type="text"
                class="px-1.5 py-0.5 text-xs font-semibold rounded border border-iolite bg-[var(--v2-card)] text-[var(--v2-ink)] outline-none"
                @keydown.enter="commitRenameSection(sec.id)"
                @keydown.esc="editingSectionId = null"
                @blur="commitRenameSection(sec.id)"
                @click.stop
              />
              <span
                v-else
                class="text-xs font-semibold text-[var(--v2-ink)] truncate"
                title="Double-click to rename section"
                @dblclick.stop="startRenameSection(sec)"
              >
                {{ sec.title }}
              </span>
              <span class="text-[10px] num px-1.5 py-0.5 rounded-full bg-[var(--v2-panel)] text-[var(--v2-ink-3)]">
                {{ filteredSectionCards(sec).length }}
              </span>
              <span
                v-if="dropTargetSectionId === sec.id"
                class="text-[10px] font-medium text-iolite dark:text-iolite-light"
              >
                Drop panel here
              </span>
            </div>

            <div class="flex items-center gap-1" @click.stop>
              <button
                class="btn-ghost px-1.5 py-0.5 text-[11px] flex items-center gap-0.5"
                title="Add panel to this section"
                @click="openAddPanel(sec.id)"
              >
                <span class="i-carbon-add text-xs" />
                Panel
              </button>
              <button
                v-if="activeBoardTab.sections.length > 1"
                class="w-5 h-5 rounded flex items-center justify-center text-[var(--v2-ink-3)] hover:text-amber-shadow dark:hover:text-amber-light hover:bg-[var(--v2-panel)] cursor-pointer"
                title="Delete section (keeps panels in first section)"
                @click="boardStore.removeSection(boardScopeKey, activeBoardTab.id, sec.id)"
              >
                <span class="i-carbon-trash-can text-[11px]" />
              </button>
            </div>
          </div>

          <!-- Section Body: Flex-wrap Multimodal Cards -->
          <div v-show="!sec.collapsed" class="p-2.5">
            <div
              v-if="filteredSectionCards(sec).length > 0"
              class="flex flex-wrap gap-2.5 items-start"
            >
              <BoardCard
                v-for="card in filteredSectionCards(sec)"
                :key="card.id"
                :scope-key="boardScopeKey"
                :scope-id="scopeId"
                :project="project"
                :tab-id="activeBoardTab.id"
                :section-id="sec.id"
                :card="card"
                :runs="visibleRuns"
                :available-metrics="catalog.scalars"
                :all-tabs="boardState.tabs"
                :tab-settings="activeBoardTab.settings"
                :hover-sync="hoverSync"
                @update:card="
                  ({ sectionId, cardId, patch, syncAll }) =>
                    boardStore.updateCard(boardScopeKey, activeBoardTab.id, sectionId, cardId, patch, {
                      syncAll,
                    })
                "
                @remove="
                  ({ sectionId, cardId }) =>
                    boardStore.removeCard(boardScopeKey, activeBoardTab.id, sectionId, cardId)
                "
                @move-to-tab="
                  ({ sectionId, cardId, targetTabId }) =>
                    boardStore.moveCard(
                      boardScopeKey,
                      activeBoardTab.id,
                      sectionId,
                      cardId,
                      targetTabId,
                    )
                "
                @drop-on-card="onCardDropOnCard"
              />
            </div>

            <div
              v-else
              class="py-6 text-center text-xs text-[var(--v2-ink-3)] border border-dashed kt-v2-line rounded-lg"
            >
              Empty section — drag any panel here or click <strong>+ Panel</strong>.
            </div>
          </div>
        </section>

        <!-- Helpful hint footer -->
        <div class="text-[10px] text-[var(--v2-ink-3)] text-center pb-2">
          Drag panel headers to reorder, move across sections, or drop onto a Tab · Drag edges to resize (Shift+drag syncs all) · Hold Shift while scrubbing sliders to sync steps
        </div>
      </div>
    </template>

    <!-- + Add Tab Modal -->
    <div
      v-if="addTabOpen"
      class="fixed inset-0 bg-black/45 z-50 flex items-center justify-center p-4"
      @click="addTabOpen = false"
    >
      <div class="kt-v2-float w-full max-w-sm p-4 space-y-3" data-testid="add-tab-modal" @click.stop>
        <div class="flex items-center justify-between">
          <span class="text-sm font-semibold">Create Custom Tab</span>
          <button class="btn-ghost" @click="addTabOpen = false">✕</button>
        </div>
        <input
          v-model="newTabName"
          type="text"
          placeholder="e.g. Overview, Loss + Samples, Paper..."
          class="w-full px-2.5 py-1.5 text-xs rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] outline-none focus:border-iolite"
          @keydown.enter="confirmAddTab"
        />
        <div class="flex justify-end gap-2">
          <button class="btn-secondary" @click="addTabOpen = false">Cancel</button>
          <button class="btn-primary" @click="confirmAddTab">Create Tab</button>
        </div>
      </div>
    </div>

    <!-- + Add Section Modal -->
    <div
      v-if="addSectionOpen"
      class="fixed inset-0 bg-black/45 z-50 flex items-center justify-center p-4"
      @click="addSectionOpen = false"
    >
      <div class="kt-v2-float w-full max-w-sm p-4 space-y-3" data-testid="add-section-modal" @click.stop>
        <div class="flex items-center justify-between">
          <span class="text-sm font-semibold">Add Collapsible Section</span>
          <button class="btn-ghost" @click="addSectionOpen = false">✕</button>
        </div>
        <input
          v-model="newSectionTitle"
          type="text"
          placeholder="e.g. Core Metrics, Qualitative Samples..."
          class="w-full px-2.5 py-1.5 text-xs rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] outline-none focus:border-iolite"
          @keydown.enter="confirmAddSection"
        />
        <div class="flex justify-end gap-2">
          <button class="btn-secondary" @click="addSectionOpen = false">Cancel</button>
          <button class="btn-primary" @click="confirmAddSection">Add Section</button>
        </div>
      </div>
    </div>

    <!-- + Add Multimodal Panel Modal -->
    <div
      v-if="addPanelOpen"
      class="fixed inset-0 bg-black/45 z-50 flex items-center justify-center p-4"
      @click="addPanelOpen = false"
    >
      <div class="kt-v2-float w-full max-w-md p-4 space-y-3" data-testid="add-panel-modal" @click.stop>
        <div class="flex items-center justify-between">
          <span class="text-sm font-semibold">Add Panel (Any Modality)</span>
          <button class="btn-ghost" @click="addPanelOpen = false">✕</button>
        </div>

        <!-- Modality selector pills -->
        <div class="grid grid-cols-4 gap-1.5">
          <button
            v-for="k in [
              { id: 'scalar', label: 'Scalar', icon: 'i-carbon-chart-line' },
              { id: 'media', label: 'Media', icon: 'i-carbon-image' },
              { id: 'distribution', label: 'Hist / KDE', icon: 'i-carbon-chart-histogram' },
              { id: 'table', label: 'Table', icon: 'i-carbon-data-table' },
            ]"
            :key="k.id"
            class="px-2 py-1.5 rounded-lg text-xs font-medium border flex items-center justify-center gap-1 cursor-pointer"
            :class="
              newPanelKind === k.id
                ? 'bg-iolite text-white border-iolite'
                : 'bg-[var(--v2-card)] kt-v2-line text-[var(--v2-ink-2)]'
            "
            @click="onPanelKindChange(k.id)"
          >
            <span :class="k.icon" class="text-xs" />
            {{ k.label }}
          </button>
        </div>

        <input
          v-model="newPanelTitle"
          type="text"
          placeholder="Custom card title (optional)..."
          class="w-full px-2.5 py-1.5 text-xs rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] outline-none focus:border-iolite"
        />

        <!-- Target Section picker -->
        <div v-if="activeBoardTab && activeBoardTab.sections.length > 1" class="flex items-center gap-2 text-xs">
          <span class="text-[var(--v2-ink-2)] shrink-0">Section:</span>
          <select
            v-model="newPanelTargetSectionId"
            class="flex-1 px-2 py-1 rounded-md border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)]"
          >
            <option v-for="s in activeBoardTab.sections" :key="s.id" :value="s.id">
              {{ s.title }}
            </option>
          </select>
        </div>

        <!-- Scalar multi-metric picker -->
        <div v-if="newPanelKind === 'scalar'" class="space-y-1.5">
          <div class="text-xs text-[var(--v2-ink-2)]">Select one or more scalar metrics to overlay:</div>
          <div class="flex flex-wrap gap-1.5 max-h-44 overflow-y-auto p-1">
            <button
              v-for="m in catalog.scalars"
              :key="m"
              class="px-2 py-1 rounded text-xs num border cursor-pointer"
              :class="
                newPanelMetrics.includes(m)
                  ? 'bg-iolite text-white border-iolite'
                  : 'bg-[var(--v2-card)] kt-v2-line text-[var(--v2-ink-2)]'
              "
              @click="toggleNewMetric(m)"
            >
              {{ m }}
            </button>
            <span v-if="catalog.scalars.length === 0" class="text-xs text-[var(--v2-ink-3)]">
              No scalar metrics logged.
            </span>
          </div>
        </div>

        <!-- Media picker -->
        <div v-else-if="newPanelKind === 'media'" class="space-y-1.5">
          <div class="text-xs text-[var(--v2-ink-2)]">Select media key:</div>
          <select
            v-if="catalog.media.length > 0"
            v-model="newPanelSingleKey"
            class="w-full px-2 py-1.5 text-xs rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)]"
          >
            <option v-for="m in catalog.media" :key="m" :value="m">{{ m }}</option>
          </select>
          <div v-else class="text-xs text-[var(--v2-ink-3)]">No media logged yet.</div>
        </div>

        <!-- Distribution picker -->
        <div v-else-if="newPanelKind === 'distribution'" class="space-y-1.5">
          <div class="text-xs text-[var(--v2-ink-2)]">Select histogram or KDE:</div>
          <select
            v-if="distributionOptions.length > 0"
            v-model="newPanelSingleKey"
            class="w-full px-2 py-1.5 text-xs rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)]"
          >
            <option v-for="d in distributionOptions" :key="d.key" :value="d.key">
              [{{ d.kind }}] {{ d.name }}
            </option>
          </select>
          <div v-else class="text-xs text-[var(--v2-ink-3)]">No histograms or KDEs logged yet.</div>
        </div>

        <!-- Table picker -->
        <div v-else-if="newPanelKind === 'table'" class="space-y-1.5">
          <div class="text-xs text-[var(--v2-ink-2)]">Select table key:</div>
          <select
            v-if="catalog.tables.length > 0"
            v-model="newPanelSingleKey"
            class="w-full px-2 py-1.5 text-xs rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)]"
          >
            <option v-for="t in catalog.tables" :key="t" :value="t">{{ t }}</option>
          </select>
          <div v-else class="text-xs text-[var(--v2-ink-3)]">No tables logged yet.</div>
        </div>

        <div class="flex justify-end gap-2 pt-2 border-t kt-v2-line">
          <button class="btn-secondary" @click="addPanelOpen = false">Cancel</button>
          <button
            class="btn-primary"
            :disabled="
              newPanelKind === 'scalar' ? newPanelMetrics.length === 0 : !newPanelSingleKey
            "
            @click="confirmAddPanel"
          >
            Add Panel
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
