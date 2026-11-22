<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue"
import uPlot from "uplot"
import "uplot/dist/uPlot.min.css"
import { useThemeStore } from "@/stores/theme"
import { api } from "@/utils/api"
import { GEM_COLORS, chartChrome } from "@/utils/palette"
import { formatNumber } from "@/utils/format"
import { useSliderSync } from "@/composables/useSliderSync"

const props = defineProps({
  scopeId: { type: String, required: true },
  project: { type: String, required: true },
  card: { type: Object, required: true },
  runs: { type: Array, required: true }, // [{ run_id, name, label, color }]
  tabSettings: { type: Object, required: true },
  hoverSync: { type: Boolean, default: true },
  height: { type: Number, default: 260 },
})

const emit = defineEmits(["patch-card"])
const theme = useThemeStore()

const loading = ref(false)
const activeRunId = ref("")
// Ascending sorted records [{ step, global_step, counts?/density?, min_val/range_min, max_val/range_max }]
const records = ref([])
const stageEl = ref(null)
const singleCanvas = ref(null)
const flowHost = ref(null)
const flowHover = ref(null)

let flowPlot = null
let resizeObs = null

const mode = computed(() => props.card.histogramMode || props.tabSettings.histogramMode || "single")
const normalize = computed(() => props.card.normalize || "per-step")
const autoAdvance = computed(() => props.card.autoAdvance !== false)

watch(
  () => props.runs,
  (list) => {
    if (list.length === 0) {
      activeRunId.value = ""
    } else if (!list.some((r) => r.run_id === activeRunId.value)) {
      activeRunId.value = list[0].run_id
    }
  },
  { immediate: true },
)

async function loadDist() {
  if (!activeRunId.value || !props.card.distName) {
    records.value = []
    return
  }
  loading.value = true
  try {
    const page = await api.distributionsPage(
      props.project,
      activeRunId.value,
      props.card.distKind || "histogram",
      props.card.distName,
      { limit: 256 },
    )
    // API returns newest-first; sort ascending for oldest -> newest slider & flow X-axis
    const sorted = [...(page.items || [])].sort((a, b) => a.step - b.step)
    records.value = sorted
  } catch {
    records.value = []
  } finally {
    loading.value = false
    await nextTick()
    renderActive()
  }
}

const steps = computed(() => records.value.map((r) => r.step))

const stepIndex = computed({
  get() {
    if (steps.value.length === 0) return 0
    if (props.card.currentStep === null || props.card.currentStep === undefined) {
      return steps.value.length - 1
    }
    const exact = steps.value.indexOf(props.card.currentStep)
    if (exact >= 0) return exact
    if (autoAdvance.value) return steps.value.length - 1
    let best = 0
    let bestDiff = Math.abs(steps.value[0] - props.card.currentStep)
    for (let i = 1; i < steps.value.length; i++) {
      const d = Math.abs(steps.value[i] - props.card.currentStep)
      if (d < bestDiff) {
        bestDiff = d
        best = i
      }
    }
    return best
  },
  set(idx) {
    const clamped = Math.max(0, Math.min(steps.value.length - 1, idx))
    const newStep = steps.value[clamped]
    if (newStep === undefined) return
    emit("patch-card", {
      currentStep: newStep,
      autoAdvance: clamped === steps.value.length - 1,
    })
    triggerSync(newStep)
  },
})

const currentRecord = computed(() => records.value[stepIndex.value] || null)

const { isShiftPressed, triggerSync } = useSliderSync(
  computed(() => `dist:${props.card.id}`),
  steps,
  (syncedStep) => {
    emit("patch-card", { currentStep: syncedStep, autoAdvance: false })
  },
)

// Viridis-inspired 5-stop colormap (clean on both Warm Stone & Warm Obsidian)
const COLOR_STOPS = [
  [36, 22, 68],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
]

function sampleColor(t, alpha = 1) {
  const c = Math.max(0, Math.min(1, t))
  const scaled = c * (COLOR_STOPS.length - 1)
  const i0 = Math.floor(scaled)
  const i1 = Math.min(COLOR_STOPS.length - 1, i0 + 1)
  const f = scaled - i0
  const r = Math.round(COLOR_STOPS[i0][0] + (COLOR_STOPS[i1][0] - COLOR_STOPS[i0][0]) * f)
  const g = Math.round(COLOR_STOPS[i0][1] + (COLOR_STOPS[i1][1] - COLOR_STOPS[i0][1]) * f)
  const b = Math.round(COLOR_STOPS[i0][2] + (COLOR_STOPS[i1][2] - COLOR_STOPS[i0][2]) * f)
  return `rgba(${r},${g},${b},${alpha})`
}

function getRecordValues(rec) {
  if (Array.isArray(rec.counts)) {
    return { vals: rec.counts, minV: rec.min_val ?? 0, maxV: rec.max_val ?? 1, isHist: true }
  }
  if (Array.isArray(rec.density)) {
    return { vals: rec.density, minV: rec.range_min ?? 0, maxV: rec.range_max ?? 1, isHist: false }
  }
  return { vals: [], minV: 0, maxV: 1, isHist: true }
}

function drawSingle() {
  const cv = singleCanvas.value
  const rec = currentRecord.value
  if (!cv || !rec) return
  const { vals, minV, maxV, isHist } = getRecordValues(rec)
  if (vals.length === 0) return

  const dpr = window.devicePixelRatio || 1
  const w = Math.max(180, cv.clientWidth || 320)
  const h = Math.max(120, cv.clientHeight || 180)
  cv.width = Math.floor(w * dpr)
  cv.height = Math.floor(h * dpr)
  const ctx = cv.getContext("2d")
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, w, h)

  const chrome = chartChrome(theme.isDark)
  const pad = { l: 44, r: 12, t: 12, b: 22 }
  const pw = w - pad.l - pad.r
  const ph = h - pad.t - pad.b

  // Baseline + grid
  ctx.strokeStyle = chrome.grid
  ctx.lineWidth = 1
  for (let g = 0; g <= 3; g++) {
    const gy = pad.t + (ph * g) / 3
    ctx.beginPath()
    ctx.moveTo(pad.l, gy)
    ctx.lineTo(pad.l + pw, gy)
    ctx.stroke()
  }

  const maxY = Math.max(1e-9, ...vals)

  if (isHist) {
    const barW = pw / vals.length
    ctx.fillStyle = `${GEM_COLORS.taaffeite}cc`
    for (let i = 0; i < vals.length; i++) {
      const bh = (vals[i] / maxY) * ph
      ctx.fillRect(pad.l + i * barW + 0.5, pad.t + ph - bh, Math.max(1, barW - 1), bh)
    }
  } else {
    ctx.beginPath()
    ctx.moveTo(pad.l, pad.t + ph)
    for (let i = 0; i < vals.length; i++) {
      const x = pad.l + (i / Math.max(1, vals.length - 1)) * pw
      const y = pad.t + ph - (vals[i] / maxY) * ph
      ctx.lineTo(x, y)
    }
    ctx.lineTo(pad.l + pw, pad.t + ph)
    ctx.closePath()
    ctx.fillStyle = `${GEM_COLORS.taaffeite}44`
    ctx.fill()
    ctx.strokeStyle = GEM_COLORS.taaffeite
    ctx.lineWidth = 2
    ctx.stroke()
  }

  ctx.fillStyle = chrome.axis
  ctx.font = "10px var(--font-mono)"
  ctx.fillText(formatNumber(minV, 3), pad.l, h - 6)
  ctx.fillText(formatNumber((minV + maxV) / 2, 3), pad.l + pw / 2 - 14, h - 6)
  ctx.fillText(formatNumber(maxV, 3), pad.l + pw - 36, h - 6)
  ctx.fillText(isHist ? String(Math.round(maxY)) : formatNumber(maxY, 3), 4, pad.t + 9)
  ctx.fillText("0", 4, pad.t + ph)
}

function destroyFlow() {
  if (flowPlot) {
    flowPlot.destroy()
    flowPlot = null
  }
}

/**
 * Render 2D Distribution Flow Heatmap using uPlot's Canvas2D draw hook!
 * X-axis = step (or global_step), Y-axis = distribution value range, Color = density/count.
 */
function drawFlowHeatmap() {
  destroyFlow()
  if (!flowHost.value || records.value.length === 0) return

  const recs = records.value
  const xAxisKey = props.tabSettings.xAxis === "step" ? "step" : "global_step"
  const xs = new Float64Array(recs.length)
  let yMin = Number.POSITIVE_INFINITY
  let yMax = Number.NEGATIVE_INFINITY
  let globalPeak = 1e-9

  const parsed = recs.map((r, i) => {
    xs[i] = r[xAxisKey] ?? r.step ?? i
    const info = getRecordValues(r)
    if (info.minV < yMin) yMin = info.minV
    if (info.maxV > yMax) yMax = info.maxV
    let localPeak = 1e-9
    for (let j = 0; j < info.vals.length; j++) {
      if (info.vals[j] > localPeak) localPeak = info.vals[j]
      if (info.vals[j] > globalPeak) globalPeak = info.vals[j]
    }
    return { ...info, localPeak, step: r.step }
  })

  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin === yMax) {
    yMin = 0
    yMax = 1
  }

  // Dummy Y envelope series so uPlot auto-scales Y to [yMin, yMax] cleanly
  const yLow = new Float64Array(recs.length).fill(yMin)
  const yHigh = new Float64Array(recs.length).fill(yMax)

  const rect = flowHost.value.getBoundingClientRect()
  const w = Math.max(180, Math.floor(rect.width || 320))
  const h = Math.max(120, Math.floor(rect.height || props.height - 62))
  const chrome = chartChrome(theme.isDark)
  const normMode = normalize.value

  const opts = {
    width: w,
    height: h,
    cursor: {
      sync: props.hoverSync ? { key: props.scopeId } : undefined,
      drag: { x: true, y: false, setScale: true },
    },
    scales: {
      x: { time: false },
      y: { range: [yMin, yMax] },
    },
    axes: [
      {
        stroke: chrome.axis,
        grid: { stroke: chrome.grid, width: 1 },
        ticks: { stroke: chrome.ticks, width: 1, size: 4 },
        font: "10px var(--font-mono)",
        size: 22,
      },
      {
        stroke: chrome.axis,
        grid: { stroke: chrome.grid, width: 1 },
        ticks: { stroke: chrome.ticks, width: 1, size: 4 },
        font: "10px var(--font-mono)",
        size: 48,
        values: (_, splits) => splits.map((v) => formatNumber(v, 3)),
      },
    ],
    series: [
      { label: xAxisKey },
      { label: "min", show: false, stroke: "transparent" },
      { label: "max", show: false, stroke: "transparent" },
    ],
    hooks: {
      draw: [
        (u) => {
          const ctx = u.ctx
          const { left, top, width: bw, height: bh } = u.bbox
          ctx.save()
          ctx.beginPath()
          ctx.rect(left, top, bw, bh)
          ctx.clip()

          const n = parsed.length
          for (let i = 0; i < n; i++) {
            const p = parsed[i]
            if (p.vals.length === 0) continue

            const cx = u.valToPos(xs[i], "x", true)
            let x0
            let x1
            if (n === 1) {
              x0 = left
              x1 = left + bw
            } else {
              const prevX = i > 0 ? u.valToPos(xs[i - 1], "x", true) : cx - (u.valToPos(xs[1], "x", true) - cx)
              const nextX = i < n - 1 ? u.valToPos(xs[i + 1], "x", true) : cx + (cx - u.valToPos(xs[n - 2], "x", true))
              x0 = (prevX + cx) / 2
              x1 = (cx + nextX) / 2
            }
            const colW = Math.max(2, x1 - x0 + 0.6)
            if (x1 < left || x0 > left + bw) continue

            const bins = p.vals.length
            const span = p.maxV - p.minV || 1
            const denom = normMode === "per-step" ? p.localPeak : normMode === "global" ? globalPeak : globalPeak

            for (let b = 0; b < bins; b++) {
              const v = p.vals[b]
              if (v <= 0) continue
              const norm = Math.min(1, v / denom)
              const v0 = p.minV + (b / bins) * span
              const v1 = p.minV + ((b + 1) / bins) * span
              const py1 = u.valToPos(v0, "y", true)
              const py0 = u.valToPos(v1, "y", true)
              const cellH = Math.max(1.5, Math.abs(py1 - py0) + 0.5)
              ctx.fillStyle = sampleColor(norm, 0.92)
              ctx.fillRect(x0, Math.min(py0, py1), colW, cellH)
            }
          }
          ctx.restore()
        },
      ],
      setCursor: [
        (u) => {
          const idx = u.cursor.idx
          if (idx === null || idx === undefined || !parsed[idx]) {
            flowHover.value = null
            return
          }
          const p = parsed[idx]
          flowHover.value = {
            step: p.step,
            xVal: xs[idx],
            minV: p.minV,
            maxV: p.maxV,
          }
        },
      ],
    },
  }

  flowPlot = new uPlot(opts, [xs, yLow, yHigh], flowHost.value)
}

function renderActive() {
  if (mode.value === "flow") {
    drawFlowHeatmap()
  } else {
    destroyFlow()
    drawSingle()
  }
}

onMounted(() => {
  resizeObs = new ResizeObserver(() => {
    nextTick(renderActive)
  })
  if (stageEl.value) resizeObs.observe(stageEl.value)
  nextTick(renderActive)
})

onBeforeUnmount(() => {
  if (resizeObs) resizeObs.disconnect()
  destroyFlow()
})

watch([() => props.project, () => props.card.distName, () => props.card.distKind, activeRunId], loadDist, {
  immediate: true,
})
watch(
  [mode, normalize, stepIndex, () => theme.isDark, () => props.height, () => props.tabSettings.xAxis, () => props.hoverSync],
  () => nextTick(renderActive),
)

defineExpose({ refresh: loadDist })
</script>

<template>
  <div class="flex flex-col h-full w-full overflow-hidden">
    <!-- Compact mode + multi-run selector bar -->
    <div class="px-2 py-1 border-b kt-v2-line/60 flex items-center justify-between gap-2 text-[10px] shrink-0">
      <!-- Multi-run pill selector (shown when comparing > 1 run) -->
      <div v-if="runs.length > 1" class="flex items-center gap-1 overflow-x-auto min-w-0">
        <button
          v-for="r in runs.slice(0, 6)"
          :key="r.run_id"
          class="px-1.5 py-0.5 rounded-full flex items-center gap-1 cursor-pointer shrink-0 border"
          :class="
            activeRunId === r.run_id
              ? 'bg-taaffeite text-white border-taaffeite'
              : 'bg-[var(--v2-panel)] text-[var(--v2-ink-2)] kt-v2-line'
          "
          @click="activeRunId = r.run_id"
        >
          <span class="w-1.5 h-1.5 rounded-full" :style="{ background: activeRunId === r.run_id ? '#fff' : r.color }" />
          <span class="truncate max-w-20">{{ r.label || r.name }}</span>
        </button>
      </div>
      <span v-else class="text-[var(--v2-ink-3)] uppercase tracking-wide">
        {{ card.distKind || "histogram" }}
      </span>

      <div class="flex items-center gap-1 shrink-0 ml-auto">
        <button
          class="px-1.5 py-0.5 rounded cursor-pointer border"
          :class="
            mode === 'single'
              ? 'bg-taaffeite text-white border-taaffeite'
              : 'kt-v2-line text-[var(--v2-ink-2)]'
          "
          @click="emit('patch-card', { histogramMode: 'single' })"
        >
          Single
        </button>
        <button
          class="px-1.5 py-0.5 rounded cursor-pointer border"
          :class="
            mode === 'flow'
              ? 'bg-taaffeite text-white border-taaffeite'
              : 'kt-v2-line text-[var(--v2-ink-2)]'
          "
          @click="emit('patch-card', { histogramMode: 'flow' })"
        >
          Flow
        </button>
        <select
          v-if="mode === 'flow'"
          :value="normalize"
          class="px-1 py-0.5 rounded border kt-v2-line bg-[var(--v2-card)] text-[var(--v2-ink)] text-[10px]"
          title="Heatmap color normalization"
          @change="emit('patch-card', { normalize: $event.target.value })"
        >
          <option value="per-step">per-step</option>
          <option value="global">global</option>
        </select>
      </div>
    </div>

    <!-- Canvas / uPlot Flow area -->
    <div ref="stageEl" class="flex-1 min-h-0 relative px-1 py-1 overflow-hidden">
      <div
        v-if="isShiftPressed && mode === 'single'"
        class="absolute top-1.5 right-2 z-10 px-1.5 py-0.5 rounded bg-iolite text-white text-[10px] font-semibold"
      >
        SYNC
      </div>

      <template v-if="records.length > 0">
        <div v-if="mode === 'flow'" ref="flowHost" class="w-full h-full" data-testid="dist-flow-plot" />
        <canvas v-else ref="singleCanvas" class="w-full h-full block" data-testid="dist-single-canvas" />
      </template>

      <div v-else class="h-full flex items-center justify-center text-xs text-[var(--v2-ink-3)]">
        {{ loading ? "Loading distribution…" : "No distribution snapshots for this run." }}
      </div>
    </div>

    <!-- Footer: Single mode slider OR Flow hover info -->
    <div
      v-if="records.length > 0"
      class="px-2.5 py-1 border-t kt-v2-line/70 flex items-center gap-2 text-[11px] shrink-0 min-h-[26px]"
    >
      <template v-if="mode === 'single' && currentRecord">
        <span class="num text-[var(--v2-ink-2)] shrink-0">step {{ currentRecord.step }}</span>
        <input
          v-if="records.length > 1"
          v-model.number="stepIndex"
          type="range"
          min="0"
          :max="records.length - 1"
          class="flex-1 min-w-0 accent-taaffeite h-1.5 cursor-pointer"
          title="Scrub step (hold Shift to sync all sliders)"
        />
        <span v-else class="flex-1" />
        <span class="num text-[10px] text-[var(--v2-ink-3)] shrink-0">
          {{ stepIndex + 1 }}/{{ records.length }}
        </span>
      </template>
      <template v-else>
        <span v-if="flowHover" class="num text-[var(--v2-ink-2)]">
          step {{ flowHover.step }} · range [{{ formatNumber(flowHover.minV, 3) }},
          {{ formatNumber(flowHover.maxV, 3) }}]
        </span>
        <span v-else class="text-[10px] text-[var(--v2-ink-3)]">
          2D flow heatmap · {{ records.length }} snapshots (drag X to zoom)
        </span>
      </template>
    </div>
  </div>
</template>
