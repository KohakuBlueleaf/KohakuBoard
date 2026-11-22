<script setup>
import { computed, nextTick, ref, watch } from "vue"
import { WIDTH_SNAPS, nearestWidthSnap, useBoardLayoutStore, widthSnapLabel } from "@/stores/boardLayout"
import ScalarCardBody from "./ScalarCardBody.vue"
import MediaCardBody from "./MediaCardBody.vue"
import DistributionCardBody from "./DistributionCardBody.vue"
import TableCardBody from "./TableCardBody.vue"

const props = defineProps({
  scopeKey: { type: String, required: true },
  scopeId: { type: String, required: true },
  project: { type: String, required: true },
  tabId: { type: String, required: true },
  sectionId: { type: String, required: true },
  card: { type: Object, required: true },
  runs: { type: Array, required: true },
  availableMetrics: { type: Array, default: () => [] },
  allTabs: { type: Array, default: () => [] },
  tabSettings: { type: Object, required: true },
  hoverSync: { type: Boolean, default: true },
})

const emit = defineEmits(["update:card", "remove", "move-to-tab", "drop-on-card"])

const boardStore = useBoardLayoutStore()

const cardEl = ref(null)
const bodyRef = ref(null)

const localWidth = ref(props.card.widthPercent || 33.333)
const localHeight = ref(props.card.height || 260)
const isResizingW = ref(false)
const isResizingH = ref(false)
const previewSnapLabel = ref("")
const isDropTarget = ref(false)

const isEditingTitle = ref(false)
const titleDraft = ref(props.card.title)
const titleInputEl = ref(null)
const settingsOpen = ref(false)

watch(
  () => props.card.widthPercent,
  (w) => {
    if (!isResizingW.value && typeof w === "number") localWidth.value = w
  },
)
watch(
  () => props.card.height,
  (h) => {
    if (!isResizingH.value && typeof h === "number") localHeight.value = h
  },
)
watch(
  () => props.card.title,
  (t) => {
    if (!isEditingTitle.value) titleDraft.value = t
  },
)

const kindMeta = computed(() => {
  switch (props.card.kind) {
    case "media":
      return { icon: "i-carbon-image", color: "text-taaffeite-shadow dark:text-taaffeite-light", label: "Media" }
    case "distribution":
      return {
        icon: "i-carbon-chart-histogram",
        color: "text-taaffeite-shadow dark:text-taaffeite-light",
        label: props.card.distKind === "kde" ? "KDE" : "Hist",
      }
    case "table":
      return { icon: "i-carbon-data-table", color: "text-sapphire dark:text-sapphire-light", label: "Table" }
    default:
      return { icon: "i-carbon-chart-line", color: "text-iolite dark:text-iolite-light", label: "Scalar" }
  }
})

const cardWidthStyle = computed(() => ({
  flex: `0 0 calc(${localWidth.value}% - 10px)`,
  maxWidth: `calc(${localWidth.value}% - 10px)`,
  minWidth: "240px",
}))

function patchCard(patch, syncAll = false) {
  emit("update:card", {
    sectionId: props.sectionId,
    cardId: props.card.id,
    patch,
    syncAll,
  })
}

function startTitleEdit() {
  titleDraft.value = props.card.title
  isEditingTitle.value = true
  nextTick(() => {
    titleInputEl.value?.focus()
    titleInputEl.value?.select()
  })
}

function commitTitle() {
  const clean = titleDraft.value.trim()
  isEditingTitle.value = false
  if (clean && clean !== props.card.title) {
    patchCard({ title: clean })
  }
}

// ---- Independent Width & Height Resize (Shift+drag syncs all cards in active tab) ----

function startResizeRight(ev) {
  ev.preventDefault()
  ev.stopPropagation()
  const el = cardEl.value
  const parent = el?.parentElement
  if (!el || !parent) return
  const startX = ev.clientX
  const startW = el.getBoundingClientRect().width
  const parentW = Math.max(300, parent.getBoundingClientRect().width)
  const syncAll = ev.shiftKey
  isResizingW.value = true
  let snapped = nearestWidthSnap((startW / parentW) * 100)
  previewSnapLabel.value = snapped.l

  const onMove = (e) => {
    const newW = Math.max(220, startW + (e.clientX - startX))
    snapped = nearestWidthSnap((newW / parentW) * 100)
    localWidth.value = snapped.p
    previewSnapLabel.value = e.shiftKey || syncAll ? `${snapped.l} (all)` : snapped.l
  }

  const onUp = (e) => {
    isResizingW.value = false
    previewSnapLabel.value = ""
    window.removeEventListener("mousemove", onMove)
    window.removeEventListener("mouseup", onUp)
    patchCard({ widthPercent: snapped.p }, syncAll || e.shiftKey)
  }

  window.addEventListener("mousemove", onMove)
  window.addEventListener("mouseup", onUp)
}

function startResizeBottom(ev) {
  ev.preventDefault()
  ev.stopPropagation()
  const startY = ev.clientY
  const startH = localHeight.value
  const syncAll = ev.shiftKey
  isResizingH.value = true

  const onMove = (e) => {
    localHeight.value = Math.max(180, Math.min(820, Math.round(startH + (e.clientY - startY))))
  }

  const onUp = (e) => {
    isResizingH.value = false
    window.removeEventListener("mousemove", onMove)
    window.removeEventListener("mouseup", onUp)
    patchCard({ height: localHeight.value }, syncAll || e.shiftKey)
  }

  window.addEventListener("mousemove", onMove)
  window.addEventListener("mouseup", onUp)
}

function startResizeCorner(ev) {
  ev.preventDefault()
  ev.stopPropagation()
  const el = cardEl.value
  const parent = el?.parentElement
  if (!el || !parent) return
  const startX = ev.clientX
  const startY = ev.clientY
  const startW = el.getBoundingClientRect().width
  const startH = localHeight.value
  const parentW = Math.max(300, parent.getBoundingClientRect().width)
  const syncAll = ev.shiftKey
  isResizingW.value = true
  isResizingH.value = true
  let snapped = nearestWidthSnap((startW / parentW) * 100)
  previewSnapLabel.value = snapped.l

  const onMove = (e) => {
    const newW = Math.max(220, startW + (e.clientX - startX))
    snapped = nearestWidthSnap((newW / parentW) * 100)
    localWidth.value = snapped.p
    localHeight.value = Math.max(180, Math.min(820, Math.round(startH + (e.clientY - startY))))
    previewSnapLabel.value = e.shiftKey || syncAll ? `${snapped.l} · ${localHeight.value}px (all)` : `${snapped.l} · ${localHeight.value}px`
  }

  const onUp = (e) => {
    isResizingW.value = false
    isResizingH.value = false
    previewSnapLabel.value = ""
    window.removeEventListener("mousemove", onMove)
    window.removeEventListener("mouseup", onUp)
    patchCard({ widthPercent: snapped.p, height: localHeight.value }, syncAll || e.shiftKey)
  }

  window.addEventListener("mousemove", onMove)
  window.addEventListener("mouseup", onUp)
}

// ---- Drag-and-drop (reorder within section, drag across sections, or drag onto Tab pill) ----

function onDragStart(ev) {
  boardStore.draggingCard = {
    scopeKey: props.scopeKey,
    tabId: props.tabId,
    sectionId: props.sectionId,
    cardId: props.card.id,
  }
  if (ev.dataTransfer) {
    ev.dataTransfer.effectAllowed = "move"
    ev.dataTransfer.setData("text/plain", `kobo-card:${props.card.id}`)
  }
}

function onDragEnd() {
  boardStore.draggingCard = null
  isDropTarget.value = false
}

function onDragOver(ev) {
  const drag = boardStore.draggingCard
  if (!drag || drag.scopeKey !== props.scopeKey || drag.cardId === props.card.id) return
  ev.preventDefault()
  isDropTarget.value = true
}

function onDragLeave() {
  isDropTarget.value = false
}

function onDrop(ev) {
  isDropTarget.value = false
  const drag = boardStore.draggingCard
  if (!drag || drag.scopeKey !== props.scopeKey || drag.cardId === props.card.id) return
  ev.preventDefault()
  ev.stopPropagation()
  emit("drop-on-card", {
    targetSectionId: props.sectionId,
    targetCardId: props.card.id,
  })
}

function toggleCardMetric(metricName) {
  const cur = [...(props.card.metrics || [])]
  const idx = cur.indexOf(metricName)
  if (idx >= 0) {
    if (cur.length <= 1) return
    cur.splice(idx, 1)
  } else {
    cur.push(metricName)
  }
  patchCard({ metrics: cur, custom: true })
}
</script>

<template>
  <div
    ref="cardEl"
    class="kt-v2-card relative flex flex-col overflow-visible transition-shadow"
    :class="[
      isDropTarget ? 'ring-2 ring-iolite border-iolite' : '',
      isResizingW || isResizingH ? 'select-none ring-1 ring-iolite/60 z-20' : '',
    ]"
    :style="cardWidthStyle"
    :data-testid="`board-card-${card.kind}`"
    :data-card-id="card.id"
    @dragover="onDragOver"
    @dragleave="onDragLeave"
    @drop="onDrop"
  >
    <!-- Live width/height snap overlay -->
    <div
      v-if="isResizingW"
      class="absolute inset-0 z-30 pointer-events-none rounded-xl border-2 border-dashed border-iolite bg-iolite/8 flex items-center justify-center"
    >
      <span class="px-2.5 py-1 rounded-md bg-iolite text-white text-xs font-semibold num shadow">
        {{ previewSnapLabel }}
      </span>
    </div>

    <!-- Card Header (draggable onto other cards, sections, OR board tabs!) -->
    <div
      class="px-2.5 py-1.5 border-b kt-v2-line flex items-center justify-between gap-1.5 shrink-0 select-none cursor-grab active:cursor-grabbing"
      draggable="true"
      title="Drag to reorder, drop on another section, or drop onto a Tab to move"
      @dragstart="onDragStart"
      @dragend="onDragEnd"
    >
      <div class="flex items-center gap-1.5 min-w-0 flex-1">
        <span class="i-carbon-draggable text-xs text-[var(--v2-ink-3)] shrink-0" />
        <span :class="[kindMeta.icon, kindMeta.color]" class="text-xs shrink-0" :title="kindMeta.label" />
        <input
          v-if="isEditingTitle"
          ref="titleInputEl"
          v-model="titleDraft"
          type="text"
          class="flex-1 min-w-0 px-1.5 py-0.5 text-xs font-semibold rounded border border-iolite bg-[var(--v2-canvas)] text-[var(--v2-ink)] outline-none"
          @keydown.enter="commitTitle"
          @keydown.esc="isEditingTitle = false"
          @blur="commitTitle"
          @click.stop
          @mousedown.stop
        />
        <span
          v-else
          class="text-xs font-semibold truncate"
          :title="`${card.title} (double-click to rename)`"
          @dblclick.stop="startTitleEdit"
        >
          {{ card.title }}
        </span>
      </div>

      <div class="flex items-center gap-0.5 shrink-0" @mousedown.stop @click.stop>
        <span class="text-[10px] num text-[var(--v2-ink-3)] mr-0.5 hidden sm:inline">
          {{ widthSnapLabel(localWidth) }}
        </span>
        <button
          class="w-5 h-5 rounded flex items-center justify-center text-[var(--v2-ink-3)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-panel)] cursor-pointer"
          title="Refresh card data"
          @click="bodyRef?.refresh?.()"
        >
          <span class="i-carbon-renew text-[11px]" />
        </button>
        <button
          v-if="card.kind === 'scalar'"
          class="w-5 h-5 rounded flex items-center justify-center text-[var(--v2-ink-3)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-panel)] cursor-pointer"
          title="Reset zoom"
          @click="bodyRef?.resetView?.()"
        >
          <span class="i-carbon-reset text-[11px]" />
        </button>
        <button
          class="w-5 h-5 rounded flex items-center justify-center cursor-pointer"
          :class="
            settingsOpen
              ? 'bg-iolite text-white'
              : 'text-[var(--v2-ink-3)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-panel)]'
          "
          title="Card settings & Move to Tab"
          @click="settingsOpen = !settingsOpen"
        >
          <span class="i-carbon-settings text-[11px]" />
        </button>
        <button
          class="w-5 h-5 rounded flex items-center justify-center text-[var(--v2-ink-3)] hover:text-amber-shadow dark:hover:text-amber-light hover:bg-[var(--v2-panel)] cursor-pointer"
          title="Remove card"
          @click="emit('remove', { sectionId, cardId: card.id })"
        >
          <span class="i-carbon-close text-xs" />
        </button>
      </div>
    </div>

    <!-- Per-card settings popover -->
    <div
      v-if="settingsOpen"
      class="absolute right-2 top-8 z-40 kt-v2-float w-72 p-3 text-xs space-y-2.5"
      @click.stop
      @mousedown.stop
    >
      <div class="flex items-center justify-between">
        <span class="font-semibold">{{ kindMeta.label }} Card Settings</span>
        <button class="btn-ghost px-1 py-0.5 text-[11px]" @click="settingsOpen = false">Done</button>
      </div>

      <!-- Width snap buttons -->
      <div>
        <div class="text-[10px] uppercase tracking-wider text-[var(--v2-ink-3)] mb-1">
          Width (Shift+click = all in tab)
        </div>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="s in WIDTH_SNAPS"
            :key="s.l"
            class="px-1.5 py-0.5 rounded text-[11px] num cursor-pointer border"
            :class="
              Math.abs(localWidth - s.p) < 0.8
                ? 'bg-iolite text-white border-iolite'
                : 'bg-[var(--v2-panel)] text-[var(--v2-ink-2)] kt-v2-line'
            "
            @click="
              (ev) => {
                localWidth = s.p
                patchCard({ widthPercent: s.p }, ev.shiftKey)
              }
            "
          >
            {{ s.l }}
          </button>
        </div>
      </div>

      <!-- Height slider -->
      <div class="flex items-center justify-between gap-2">
        <span class="text-[var(--v2-ink-2)]">Height</span>
        <input
          type="range"
          min="180"
          max="720"
          step="20"
          :value="localHeight"
          class="flex-1 accent-iolite"
          @input="
            (e) => {
              localHeight = Number(e.target.value)
              patchCard({ height: localHeight })
            }
          "
        />
        <span class="num text-[11px] w-10 text-right">{{ localHeight }}px</span>
      </div>

      <!-- Move to Tab dropdown -->
      <div v-if="allTabs.length > 1" class="flex items-center justify-between gap-2">
        <span class="text-[var(--v2-ink-2)]">Move to Tab</span>
        <select
          :value="tabId"
          class="px-1.5 py-0.5 rounded border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] text-[11px] max-w-36"
          @change="
            (e) => {
              if (e.target.value !== tabId) {
                emit('move-to-tab', { sectionId, cardId: card.id, targetTabId: e.target.value })
                settingsOpen = false
              }
            }
          "
        >
          <option v-for="t in allTabs" :key="t.id" :value="t.id">{{ t.name }}</option>
        </select>
      </div>

      <!-- Scalar-specific settings -->
      <template v-if="card.kind === 'scalar'">
        <div class="pt-1 border-t kt-v2-line space-y-2">
          <div class="flex items-center justify-between gap-2">
            <span class="text-[var(--v2-ink-2)]">X-Axis</span>
            <select
              :value="card.xAxis || ''"
              class="px-1.5 py-0.5 rounded border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] text-[11px] max-w-40"
              @change="patchCard({ xAxis: $event.target.value || undefined })"
            >
              <option value="">Inherit ({{ tabSettings.xAxis }})</option>
              <option value="global_step">global_step</option>
              <option value="step">step</option>
              <option value="relative_time">relative_time (s)</option>
              <option v-for="m in availableMetrics" :key="m" :value="m">{{ m }}</option>
            </select>
          </div>

          <div class="flex items-center justify-between gap-2">
            <span class="text-[var(--v2-ink-2)]">Log Y</span>
            <button
              class="px-2 py-0.5 rounded text-[11px] border cursor-pointer"
              :class="
                card.logY === true
                  ? 'bg-sapphire text-white border-sapphire'
                  : 'kt-v2-line bg-[var(--v2-panel)]'
              "
              @click="patchCard({ logY: card.logY === true ? false : true })"
            >
              {{ card.logY === undefined ? "Inherit" : card.logY ? "Log" : "Linear" }}
            </button>
          </div>

          <div class="flex items-center justify-between gap-2">
            <span class="text-[var(--v2-ink-2)]">Line Width</span>
            <input
              type="range"
              min="1"
              max="4"
              step="0.25"
              :value="card.lineWidth || 1.75"
              class="w-24 accent-iolite"
              @input="patchCard({ lineWidth: Number($event.target.value) })"
            />
            <span class="num text-[11px] w-7 text-right">{{ (card.lineWidth || 1.75).toFixed(2) }}</span>
          </div>

          <div v-if="availableMetrics.length > 1">
            <div class="text-[10px] uppercase tracking-wider text-[var(--v2-ink-3)] mb-1">
              Y Metrics (overlay)
            </div>
            <div class="max-h-24 overflow-y-auto flex flex-wrap gap-1">
              <button
                v-for="m in availableMetrics"
                :key="m"
                class="px-1.5 py-0.5 rounded text-[10px] num cursor-pointer border"
                :class="
                  (card.metrics || []).includes(m)
                    ? 'bg-iolite text-white border-iolite'
                    : 'bg-[var(--v2-panel)] text-[var(--v2-ink-2)] kt-v2-line'
                "
                @click="toggleCardMetric(m)"
              >
                {{ m }}
              </button>
            </div>
          </div>

          <button class="btn-secondary w-full text-[11px] py-1" @click="bodyRef?.exportPNG?.()">
            Export PNG
          </button>
        </div>
      </template>
    </div>

    <!-- Unified Card Body (dispatched by card.kind) -->
    <div class="w-full overflow-hidden rounded-b-xl" :style="{ height: `${localHeight}px` }">
      <ScalarCardBody
        v-if="card.kind === 'scalar'"
        ref="bodyRef"
        :scope-id="scopeId"
        :project="project"
        :card="card"
        :runs="runs"
        :tab-settings="tabSettings"
        :hover-sync="hoverSync"
        :height="localHeight"
      />
      <MediaCardBody
        v-else-if="card.kind === 'media'"
        ref="bodyRef"
        :project="project"
        :card="card"
        :runs="runs"
        :height="localHeight"
        @patch-card="(p) => patchCard(p)"
      />
      <DistributionCardBody
        v-else-if="card.kind === 'distribution'"
        ref="bodyRef"
        :scope-id="scopeId"
        :project="project"
        :card="card"
        :runs="runs"
        :tab-settings="tabSettings"
        :hover-sync="hoverSync"
        :height="localHeight"
        @patch-card="(p) => patchCard(p)"
      />
      <TableCardBody
        v-else-if="card.kind === 'table'"
        ref="bodyRef"
        :project="project"
        :card="card"
        :runs="runs"
        :height="localHeight"
        @patch-card="(p) => patchCard(p)"
      />
    </div>

    <!-- Independent resize handles (Shift+drag syncs all cards in tab) -->
    <div
      class="absolute top-0 right-0 bottom-2 w-1.5 cursor-ew-resize hover:bg-iolite/30 rounded-tr-xl z-10"
      title="Drag to resize width (Shift+drag syncs all)"
      @mousedown="startResizeRight"
    />
    <div
      class="absolute left-0 right-2 bottom-0 h-1.5 cursor-ns-resize hover:bg-iolite/30 rounded-bl-xl z-10"
      title="Drag to resize height (Shift+drag syncs all)"
      @mousedown="startResizeBottom"
    />
    <div
      class="absolute right-0 bottom-0 w-3.5 h-3.5 cursor-nwse-resize z-20 flex items-end justify-end p-0.5 opacity-45 hover:opacity-100"
      title="Drag corner to resize width & height (Shift+drag syncs all)"
      @mousedown="startResizeCorner"
    >
      <span class="w-2 h-2 border-r-2 border-b-2 border-iolite rounded-br-[2px]" />
    </div>
  </div>
</template>
