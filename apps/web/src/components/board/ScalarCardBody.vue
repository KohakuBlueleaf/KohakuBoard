<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue"
import uPlot from "uplot"
import "uplot/dist/uPlot.min.css"
import { useSeriesCacheStore } from "@/stores/seriesCache"
import { useThemeStore } from "@/stores/theme"
import { alignSeriesForUPlot, pairMetricAxes } from "@/utils/seriesBuffer"
import { RUN_PALETTE, chartChrome } from "@/utils/palette"
import { formatCount, formatNumber } from "@/utils/format"

const BUILTIN_AXES = new Set(["global_step", "step", "relative_time"])

const props = defineProps({
  scopeId: { type: String, required: true },
  project: { type: String, required: true },
  card: { type: Object, required: true },
  runs: { type: Array, required: true }, // [{ run_id, name, label, color }]
  tabSettings: { type: Object, required: true },
  hoverSync: { type: Boolean, default: true },
  height: { type: Number, default: 240 },
})

const cache = useSeriesCacheStore()
const theme = useThemeStore()

const hostEl = ref(null)
const plotHost = ref(null)
const isVisible = ref(false)
const hoverState = ref(null)

let plot = null
let resizeObs = null
let interObs = null
let userZoomed = false
let lastSeriesSig = ""

const metricsList = computed(() => props.card.metrics || [])
const effectiveXAxis = computed(() => props.card.xAxis || props.tabSettings.xAxis || "global_step")
const effectiveSmoothMode = computed(() =>
  props.card.smoothingMode !== undefined
    ? props.card.smoothingMode
    : props.tabSettings.smoothingMode || "ema",
)
const effectiveSmoothVal = computed(() =>
  props.card.smoothingValue !== undefined
    ? props.card.smoothingValue
    : Number(props.tabSettings.smoothingValue || 0),
)
const effectiveLogY = computed(() =>
  props.card.logY !== undefined ? props.card.logY : !!props.tabSettings.logY,
)
const effectiveShowOriginal = computed(() =>
  props.card.showOriginal !== undefined
    ? props.card.showOriginal
    : props.tabSettings.showOriginal !== false,
)
const effectiveShowMarkers = computed(() =>
  props.card.showMarkers !== undefined ? props.card.showMarkers : !!props.tabSettings.showMarkers,
)
const effectiveLineWidth = computed(() => props.card.lineWidth || 1.75)

const hasSmoothing = computed(
  () => effectiveSmoothMode.value !== "none" && effectiveSmoothVal.value > 0,
)

const runIds = computed(() => props.runs.map((r) => r.run_id))

const seriesDescriptors = computed(() => {
  const out = []
  const multiMetric = metricsList.value.length > 1
  let colorIdx = 0
  for (const m of metricsList.value) {
    for (const r of props.runs) {
      const color = multiMetric ? RUN_PALETTE[colorIdx % RUN_PALETTE.length] : r.color
      colorIdx++
      const rLabel = r.label || r.name || r.run_id
      out.push({
        key: `${r.run_id}::${m}`,
        runId: r.run_id,
        metric: m,
        label: multiMetric ? (props.runs.length > 1 ? `${rLabel} · ${m}` : m) : rLabel,
        color,
      })
    }
  }
  return out
})

const neededMetrics = computed(() => {
  const set = new Set(metricsList.value)
  if (!BUILTIN_AXES.has(effectiveXAxis.value)) {
    set.add(effectiveXAxis.value)
  }
  return Array.from(set)
})

const bufferStats = computed(() => {
  let loaded = 0
  let total = 0
  let hasNonFinite = false
  let complete = true
  let versionSum = 0
  const customX = !BUILTIN_AXES.has(effectiveXAxis.value) ? effectiveXAxis.value : null

  for (const desc of seriesDescriptors.value) {
    versionSum += cache.getVersion(props.project, desc.runId, desc.metric)
    if (customX) {
      versionSum += cache.getVersion(props.project, desc.runId, customX)
    }
    const buf = cache.getBuffer(props.project, desc.runId, desc.metric)
    if (!buf) {
      complete = false
      continue
    }
    loaded += buf.loadedCount
    total += buf.total
    if (buf.hasNonFinite) hasNonFinite = true
    if (!buf.complete) complete = false
  }
  return {
    loaded,
    total,
    progress: total > 0 ? Math.min(1, loaded / total) : complete ? 1 : 0,
    complete,
    hasNonFinite,
    versionSum,
  }
})

function buildAlignedData() {
  const entries = []
  const active = []
  const xAxis = effectiveXAxis.value
  const isCustomX = !BUILTIN_AXES.has(xAxis)

  for (const desc of seriesDescriptors.value) {
    const yBuf = cache.getBuffer(props.project, desc.runId, desc.metric)
    if (!yBuf || yBuf.loadedCount === 0) continue
    if (isCustomX) {
      const xBuf = cache.getBuffer(props.project, desc.runId, xAxis)
      const paired = pairMetricAxes(xBuf, yBuf)
      if (paired.x.length === 0) continue
      entries.push(paired)
    } else {
      const v = yBuf.view(xAxis)
      entries.push({ x: v.x, y: v.y })
    }
    active.push(desc)
  }
  if (entries.length === 0) {
    return { active: [], cols: [new Float64Array(0)] }
  }
  const cols = alignSeriesForUPlot(entries, {
    mode: effectiveSmoothMode.value,
    value: effectiveSmoothVal.value,
    showOriginal: effectiveShowOriginal.value,
  })
  return { active, cols }
}

function buildOptions(width, height, active) {
  const chrome = chartChrome(theme.isDark)
  const lw = effectiveLineWidth.value
  const smooth = hasSmoothing.value
  const showOrig = effectiveShowOriginal.value
  const pts = effectiveShowMarkers.value ? { show: true, size: 4 } : { show: false }

  const series = [{ label: effectiveXAxis.value }]
  for (const s of active) {
    if (smooth) {
      series.push({ label: s.label, stroke: s.color, width: lw + 0.25, points: pts })
      if (showOrig) {
        series.push({
          label: `${s.label} (raw)`,
          stroke: `${s.color}38`,
          width: 1,
          points: { show: false },
        })
      }
    } else {
      series.push({ label: s.label, stroke: s.color, width: lw, points: pts })
    }
  }

  return {
    width: Math.max(180, Math.floor(width)),
    height: Math.max(120, Math.floor(height)),
    cursor: {
      sync: props.hoverSync ? { key: props.scopeId } : undefined,
      drag: { x: true, y: false, setScale: true },
    },
    scales: {
      x: { time: false },
      y: { distr: effectiveLogY.value ? 3 : 1 },
    },
    axes: [
      {
        stroke: chrome.axis,
        grid: { stroke: chrome.grid, width: 1 },
        ticks: { stroke: chrome.ticks, width: 1, size: 4 },
        font: "11px var(--font-mono)",
        size: 24,
      },
      {
        stroke: chrome.axis,
        grid: { stroke: chrome.grid, width: 1 },
        ticks: { stroke: chrome.ticks, width: 1, size: 4 },
        font: "11px var(--font-mono)",
        size: 50,
        values: (_, splits) => splits.map((v) => formatNumber(v, 3)),
      },
    ],
    series,
    hooks: {
      setSelect: [
        (u) => {
          if (u.select.width > 4) userZoomed = true
        },
      ],
      setCursor: [
        (u) => {
          const idx = u.cursor.idx
          if (idx === null || idx === undefined || !u.data[0] || u.data[0].length === 0) {
            hoverState.value = null
            return
          }
          const xVal = u.data[0][idx]
          const rows = []
          const stride = smooth && showOrig ? 2 : 1
          for (let i = 0; i < active.length; i++) {
            const val = u.data[1 + i * stride]?.[idx]
            if (val !== null && val !== undefined) {
              rows.push({
                key: active[i].key,
                label: active[i].label,
                color: active[i].color,
                value: val,
              })
            }
          }
          hoverState.value = { xVal, rows }
        },
      ],
    },
  }
}

function renderChart() {
  if (!plotHost.value) return
  const { active, cols } = buildAlignedData()
  if (active.length === 0 || cols[0].length === 0) {
    if (plot) {
      plot.destroy()
      plot = null
      lastSeriesSig = ""
    }
    return
  }

  const sig = [
    active.map((a) => `${a.key}:${a.color}`).join("|"),
    theme.isDark,
    effectiveXAxis.value,
    effectiveSmoothMode.value,
    effectiveSmoothVal.value,
    effectiveShowOriginal.value,
    effectiveShowMarkers.value,
    effectiveLogY.value,
    effectiveLineWidth.value,
    props.hoverSync,
  ].join("~")

  const rect = plotHost.value.getBoundingClientRect()
  const w = Math.max(180, rect.width || 320)
  const h = Math.max(120, props.height - 44)

  if (!plot || sig !== lastSeriesSig) {
    if (plot) plot.destroy()
    plot = new uPlot(buildOptions(w, h, active), cols, plotHost.value)
    lastSeriesSig = sig
    userZoomed = false
  } else {
    plot.setData(cols, !userZoomed)
    if (Math.abs(plot.width - Math.floor(w)) > 2 || Math.abs(plot.height - Math.floor(h)) > 2) {
      plot.setSize({ width: Math.floor(w), height: Math.floor(h) })
    }
  }
}

function requestData() {
  if (!isVisible.value || runIds.value.length === 0 || neededMetrics.value.length === 0) return
  for (const m of neededMetrics.value) {
    cache.requestMetric(props.project, runIds.value, m)
  }
}

function resetView() {
  userZoomed = false
  if (plot) {
    const { cols } = buildAlignedData()
    if (cols[0].length > 0) plot.setData(cols, true)
  }
}

function exportPNG() {
  const cv = plotHost.value?.querySelector("canvas")
  if (!cv) return
  const a = document.createElement("a")
  a.href = cv.toDataURL("image/png")
  a.download = `${(props.card.title || "chart").replace(/[^\w.-]+/g, "_")}.png`
  a.click()
}

function refresh() {
  for (const rid of runIds.value) {
    cache.pollLiveTail(props.project, rid)
  }
  requestData()
}

defineExpose({ resetView, exportPNG, refresh })

onMounted(() => {
  interObs = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        isVisible.value = true
        requestData()
      }
    },
    { rootMargin: "220px" },
  )
  if (hostEl.value) interObs.observe(hostEl.value)

  resizeObs = new ResizeObserver(() => {
    if (!plotHost.value) return
    if (!plot) {
      renderChart()
      return
    }
    const rect = plotHost.value.getBoundingClientRect()
    const w = Math.floor(rect.width)
    const h = Math.max(120, Math.floor(props.height - 44))
    if (w > 40) plot.setSize({ width: w, height: h })
  })
  if (plotHost.value) resizeObs.observe(plotHost.value)

  // Render immediately if series data is already cached (e.g. after switching tabs!)
  nextTick(renderChart)
})

onBeforeUnmount(() => {
  if (interObs) interObs.disconnect()
  if (resizeObs) resizeObs.disconnect()
  if (plot) {
    plot.destroy()
    plot = null
  }
})

watch([runIds, neededMetrics, isVisible], requestData, { immediate: true })
watch(
  [
    () => bufferStats.value.versionSum,
    () => theme.isDark,
    effectiveXAxis,
    effectiveSmoothMode,
    effectiveSmoothVal,
    effectiveShowOriginal,
    effectiveShowMarkers,
    effectiveLogY,
    effectiveLineWidth,
    () => props.hoverSync,
    () => props.height,
    () => props.runs,
  ],
  () => nextTick(renderChart),
)
</script>

<template>
  <div ref="hostEl" class="flex flex-col h-full w-full overflow-hidden">
    <!-- Loading progress bar -->
    <div v-if="!bufferStats.complete && bufferStats.total > 0" class="h-[2px] w-full bg-[var(--v2-line)] shrink-0">
      <div
        class="h-full bg-iolite transition-all duration-150"
        :style="{ width: `${Math.round(bufferStats.progress * 100)}%` }"
      />
    </div>

    <!-- Main uPlot area -->
    <div class="relative flex-1 min-h-0 px-1 pt-1" @dblclick="resetView">
      <div ref="plotHost" class="w-full h-full" />

      <!-- Non-finite indicator -->
      <span
        v-if="bufferStats.hasNonFinite"
        class="absolute top-1.5 right-2 text-[10px] px-1.5 py-0.5 rounded bg-amber-light text-amber-shadow dark:bg-amber-shadow/50 dark:text-amber-light pointer-events-none"
        title="Contains NaN/Inf values"
      >
        NaN/Inf
      </span>

      <div
        v-if="bufferStats.loaded === 0"
        class="absolute inset-0 flex items-center justify-center text-xs text-[var(--v2-ink-3)] pointer-events-none"
      >
        {{ runs.length === 0 ? "No visible runs" : "Loading scalar data…" }}
      </div>
    </div>

    <!-- Compact hover / legend footer -->
    <div
      class="px-2.5 py-1 border-t kt-v2-line/70 flex items-center gap-2.5 text-[11px] overflow-x-auto whitespace-nowrap shrink-0 min-h-[24px]"
    >
      <template v-if="hoverState">
        <span class="num text-[var(--v2-ink-3)] shrink-0">
          {{ effectiveXAxis }}={{ formatNumber(hoverState.xVal, 2) }}
        </span>
        <span
          v-for="row in hoverState.rows.slice(0, 6)"
          :key="row.key"
          class="flex items-center gap-1 shrink-0"
        >
          <span class="w-2 h-2 rounded-full" :style="{ background: row.color }" />
          <span class="truncate max-w-28 text-[var(--v2-ink-2)]">{{ row.label }}</span>
          <span class="num font-medium">{{ formatNumber(row.value, 4) }}</span>
        </span>
      </template>
      <template v-else>
        <span
          v-for="s in seriesDescriptors.slice(0, 6)"
          :key="s.key"
          class="flex items-center gap-1 shrink-0 text-[var(--v2-ink-2)]"
        >
          <span class="w-2 h-2 rounded-full" :style="{ background: s.color }" />
          <span class="truncate max-w-28">{{ s.label }}</span>
        </span>
        <span v-if="seriesDescriptors.length > 6" class="text-[10px] text-[var(--v2-ink-3)]">
          +{{ seriesDescriptors.length - 6 }}
        </span>
        <span class="ml-auto text-[10px] num text-[var(--v2-ink-3)]">
          {{ formatCount(bufferStats.loaded) }} pts
        </span>
      </template>
    </div>
  </div>
</template>
