<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";
import HistogramSurfaceMini from "./HistogramSurfaceMini.vue";
import { useSliderSync } from "@/composables/useSliderSync";

const props = defineProps({
  histogramData: Array,
  height: Number,
  currentStep: Number,
  cardId: String,
  initialMode: {
    type: String,
    default: "single", // "single" or "flow"
  },
  downsampleRate: {
    type: Number,
    default: 1,
  },
  xAxis: {
    type: String,
    default: "global_step", // "step" or "global_step"
  },
  flowSurfaceEnabled: {
    type: Boolean,
    default: false,
  },
  surfaceData: {
    type: Object,
    default: null,
  },
  surfaceLoading: {
    type: Boolean,
    default: false,
  },
  surfaceAxis: {
    type: String,
    default: "global_step",
  },
  surfaceNormalize: {
    type: String,
    default: "none",
  },
  surfaceAspect: {
    type: Object,
    default: () => ({
      x: 1.2,
      y: 1,
      z: 0.85,
    }),
  },
});

const emit = defineEmits(["update:currentStep", "update:mode", "update:xAxis"]);

const plotDiv = ref(null);
let resizeObserver = null;
const viewMode = ref(props.initialMode);
const colorscale = ref("Viridis");
const normalize = ref("per-step"); // Default per-step for better contrast
const showSettings = ref(false);
const xAxisType = ref(props.xAxis);
const surfaceModeActive = computed(
  () => viewMode.value === "flow" && props.flowSurfaceEnabled,
);
const shouldShowSurface = computed(
  () =>
    surfaceModeActive.value &&
    props.surfaceData &&
    Array.isArray(props.surfaceData.matrix) &&
    props.surfaceData.matrix.length > 0,
);
const surfacePlotHeight = computed(() => Math.max(props.height - 140, 220));
const surfaceAxisLabel = computed(() => {
  if (props.surfaceAxis === "relative_walltime") {
    const stats = props.surfaceData?.axis_stats;
    const span =
      stats && Number.isFinite(stats.max) && Number.isFinite(stats.min)
        ? stats.max - stats.min
        : 0;
    const unit = resolveRelativeTimeUnit(span);
    return `Time (${unit.label})`;
  }
  return getAxisTitle(props.surfaceAxis || "global_step");
});
const surfaceDensityLabel = computed(() =>
  props.surfaceNormalize && props.surfaceNormalize !== "none"
    ? "Normalized Density"
    : "Density",
);
const DEFAULT_RELATIVE_UNIT = {
  scale: 1,
  label: "Seconds",
  suffix: "s",
};

function resolveRelativeTimeUnit(spanSeconds) {
  if (!Number.isFinite(spanSeconds) || spanSeconds <= 0) {
    return { ...DEFAULT_RELATIVE_UNIT };
  }
  if (spanSeconds > 150 * 60) {
    return { scale: 3600, label: "Hours", suffix: " h" };
  }
  if (spanSeconds > 150) {
    return { scale: 60, label: "Minutes", suffix: " min" };
  }
  return { ...DEFAULT_RELATIVE_UNIT };
}

function formatRelativeDurationValue(seconds, unit = DEFAULT_RELATIVE_UNIT) {
  if (!Number.isFinite(seconds)) return "N/A";
  if (unit.scale === 3600) {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${mins}m ${secs}s`;
  }
  if (unit.scale === 60) {
    const mins = seconds / 60;
    return `${mins.toFixed(2)} min`;
  }
  return `${seconds.toFixed(2)} s`;
}

function formatRelativeTickValue(seconds, unit = DEFAULT_RELATIVE_UNIT) {
  if (!Number.isFinite(seconds)) return "";
  const value = seconds / unit.scale;
  if (unit.scale === 1) {
    return `${value.toFixed(0)}${unit.suffix}`;
  }
  return `${value.toFixed(1)}${unit.suffix}`;
}

function getAxisValue(entry, axisType) {
  if (axisType === "global_step") {
    return entry.global_step ?? entry.step ?? 0;
  }
  if (axisType === "relative_walltime" && entry.relative_walltime != null) {
    return entry.relative_walltime;
  }
  return entry.step ?? 0;
}

function getAxisTitle(axisType, unit) {
  if (axisType === "global_step") return "Global Step";
  if (axisType === "relative_walltime") {
    const activeUnit = unit || DEFAULT_RELATIVE_UNIT;
    return `Time (${activeUnit.label})`;
  }
  return "Training Step";
}

function formatAxisValue(value, axisType, unit) {
  if (!Number.isFinite(value)) return "N/A";
  if (axisType === "relative_walltime") {
    return formatRelativeDurationValue(value, unit || DEFAULT_RELATIVE_UNIT);
  }
  return Math.round(value).toString();
}

function computeGlobalValueRange(entries, fallbackEdges, paddingRatio = 0.05) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  const updateRange = (edges) => {
    if (!Array.isArray(edges) || edges.length < 2) return;
    const entryMin = edges[0];
    const entryMax = edges[edges.length - 1];
    if (Number.isFinite(entryMin) && entryMin < min) {
      min = entryMin;
    }
    if (Number.isFinite(entryMax) && entryMax > max) {
      max = entryMax;
    }
  };

  entries.forEach((entry) => updateRange(entry.bins));

  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    updateRange(fallbackEdges);
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }

  if (min === max) {
    const pad = Math.max(Math.abs(min) * 0.05, 0.5);
    return { min: min - pad, max: max + pad };
  }

  const span = max - min;
  const pad = span * paddingRatio;
  return {
    min: min - pad,
    max: max + pad,
  };
}

function buildTickInfo(axisValues, axisType, unit) {
  if (!axisValues || axisValues.length === 0) {
    return { tickvals: [], ticktext: [] };
  }
  const desiredTicks = Math.min(6, axisValues.length);
  const denominator = Math.max(1, desiredTicks - 1);
  const step =
    axisValues.length > 1
      ? Math.max(1, Math.floor((axisValues.length - 1) / denominator))
      : 1;
  const tickvals = [];
  const ticktext = [];

  for (let i = 0; i < axisValues.length; i += step) {
    const value = axisValues[i];
    tickvals.push(value);
    if (axisType === "relative_walltime") {
      ticktext.push(
        formatRelativeTickValue(value, unit || DEFAULT_RELATIVE_UNIT),
      );
    } else {
      ticktext.push(Math.round(value).toString());
    }
  }

  const lastValue = axisValues[axisValues.length - 1];
  if (tickvals[tickvals.length - 1] !== lastValue) {
    tickvals.push(lastValue);
    ticktext.push(
      axisType === "relative_walltime"
        ? formatRelativeTickValue(lastValue, unit || DEFAULT_RELATIVE_UNIT)
        : Math.round(lastValue).toString(),
    );
  }

  return { tickvals, ticktext };
}

function computeAxisRange(axisValues) {
  if (!axisValues || axisValues.length === 0) {
    return [0, 1];
  }
  const min = Math.min(...axisValues);
  const max = Math.max(...axisValues);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  if (min === max) {
    return [min - 0.5, max + 0.5];
  }
  const padding = (max - min) * 0.05 || 0.5;
  return [min - padding, max + padding];
}

// Watch for prop changes and update viewMode (for global settings)
watch(
  () => props.initialMode,
  (newMode) => {
    if (newMode && newMode !== viewMode.value) {
      viewMode.value = newMode;
    }
  },
);

const stepIndex = computed({
  get() {
    if (!props.histogramData || props.histogramData.length === 0) return 0;
    return props.histogramData.findIndex(
      (item) => item.step === props.currentStep,
    );
  },
  set(index) {
    if (props.histogramData && props.histogramData[index]) {
      const newStep = props.histogramData[index].step;
      emit("update:currentStep", newStep);

      // Trigger synchronization if shift is pressed
      triggerSync(newStep);
    }
  },
});

// Setup slider synchronization
const { isShiftPressed, triggerSync } = useSliderSync(
  computed(() => `histogram-${props.cardId}`),
  computed(() => props.histogramData || []),
  (newStep) => {
    emit("update:currentStep", newStep);
  },
);

const currentHistogram = computed(() => {
  if (!props.histogramData || props.histogramData.length === 0) return null;
  const index = stepIndex.value >= 0 ? stepIndex.value : 0;
  return props.histogramData[index];
});

const histogramValueRange = computed(() => {
  if (!props.histogramData || props.histogramData.length === 0) return null;
  const fallbackEdges = props.histogramData.find(
    (entry) => Array.isArray(entry.bins) && entry.bins.length > 1,
  )?.bins;
  return computeGlobalValueRange(props.histogramData, fallbackEdges);
});

watch(
  currentHistogram,
  () => {
    if (viewMode.value === "single") createPlot();
  },
  { deep: true },
);

watch(
  () => props.height,
  () => {
    createPlot();
  },
);

watch(surfaceModeActive, (active) => {
  if (active) {
    if (plotDiv.value) {
      Plotly.purge(plotDiv.value);
    }
  } else if (viewMode.value === "flow") {
    createPlot();
  }
});

watch(
  () => props.surfaceData,
  () => {
    if (surfaceModeActive.value && plotDiv.value) {
      Plotly.purge(plotDiv.value);
    }
  },
  { deep: true },
);

watch(viewMode, (newMode) => {
  emit("update:mode", newMode); // Emit to parent to save
  createPlot();
});

watch(colorscale, () => {
  if (viewMode.value === "flow") createPlot();
});

watch(normalize, () => {
  if (viewMode.value === "flow") createPlot();
});

watch(
  () => props.downsampleRate,
  () => {
    if (viewMode.value === "flow") createPlot();
  },
);

watch(xAxisType, (newAxis) => {
  emit("update:xAxis", newAxis);
  if (viewMode.value === "flow") createPlot();
});

watch(
  () => props.xAxis,
  (newAxis) => {
    if (newAxis && newAxis !== xAxisType.value) {
      xAxisType.value = newAxis;
    }
  },
);

onMounted(() => {
  createPlot();
  setupResizeObserver();
});

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect();
  }
});

function setupResizeObserver() {
  if (!plotDiv.value) return;

  resizeObserver = new ResizeObserver(() => {
    if (plotDiv.value) {
      Plotly.Plots.resize(plotDiv.value);
    }
  });

  resizeObserver.observe(plotDiv.value.parentElement || plotDiv.value);
}

function createPlot() {
  if (surfaceModeActive.value) {
    if (plotDiv.value) {
      Plotly.purge(plotDiv.value);
    }
    return;
  }

  if (!plotDiv.value) return;

  const isDark = document.documentElement.classList.contains("dark");
  const colors = {
    text: isDark ? "#e5e7eb" : "#1f2937",
    grid: isDark ? "rgba(156, 163, 175, 0.2)" : "rgba(156, 163, 175, 0.3)",
  };

  if (viewMode.value === "flow") {
    createDistributionFlowPlot(colors);
  } else {
    createSingleStepPlot(colors);
  }
}

function createSingleStepPlot(colors) {
  if (!currentHistogram.value) return;

  // Use pre-computed histogram (bins + counts) or raw values
  const trace =
    currentHistogram.value.bins && currentHistogram.value.counts
      ? {
          type: "bar",
          x: currentHistogram.value.bins.slice(0, -1).map((bin, i) => {
            // Use bin centers for x-axis
            return (bin + currentHistogram.value.bins[i + 1]) / 2;
          }),
          y: currentHistogram.value.counts,
          marker: {
            color: document.documentElement.classList.contains("dark")
              ? "rgba(96, 165, 250, 0.7)"
              : "rgba(59, 130, 246, 0.7)",
          },
        }
      : {
          // Fallback: raw values (old format)
          type: "histogram",
          x: currentHistogram.value.values,
          nbinsx: currentHistogram.value.num_bins || 30,
          marker: {
            color: document.documentElement.classList.contains("dark")
              ? "rgba(96, 165, 250, 0.7)"
              : "rgba(59, 130, 246, 0.7)",
          },
        };

  const globalRange = histogramValueRange.value;
  const xAxisRange =
    globalRange &&
    Number.isFinite(globalRange.min) &&
    Number.isFinite(globalRange.max)
      ? [globalRange.min, globalRange.max]
      : null;

  const layout = {
    xaxis: {
      gridcolor: colors.grid,
      color: colors.text,
      title: "Value",
      ...(xAxisRange ? { range: xAxisRange } : {}),
    },
    yaxis: { gridcolor: colors.grid, color: colors.text, title: "Count" },
    height: props.height - 100,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 10, r: 20, b: 30, l: 50 },
    autosize: true,
  };

  Plotly.react(plotDiv.value, [trace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

function createDistributionFlowPlot(colors) {
  if (!props.histogramData || props.histogramData.length === 0) return;

  // Adaptive downsampling
  let rate = props.downsampleRate ?? 1;
  if (rate === -1) {
    // Target: ~200 histogram entries for smooth heatmap
    const targetEntries = 200;
    if (props.histogramData.length <= targetEntries) {
      rate = 1; // No downsampling needed
    } else {
      rate = Math.ceil(props.histogramData.length / targetEntries);
    }
  }

  const sampledData =
    rate > 1
      ? props.histogramData.filter((_, idx) => idx % rate === 0)
      : props.histogramData;

  if (!sampledData.length) return;
  const referenceEntry = sampledData.find(
    (entry) => Array.isArray(entry.bins) && entry.bins.length > 1,
  );
  if (!referenceEntry) {
    console.warn("[HistogramViewer] Flow mode requires histogram bins");
    return;
  }

  const binEdges = referenceEntry.bins;
  const binCenters = binEdges.slice(0, -1).map((bin, i) => {
    return (bin + binEdges[i + 1]) / 2;
  });
  const expectedBinCount = binEdges.length;

  const filteredEntries = [];
  const axisValues = [];
  sampledData.forEach((entry) => {
    if (
      !Array.isArray(entry.counts) ||
      entry.counts.length !== expectedBinCount - 1 ||
      !Array.isArray(entry.bins) ||
      entry.bins.length !== expectedBinCount
    ) {
      return;
    }
    filteredEntries.push(entry);
    axisValues.push(getAxisValue(entry, xAxisType.value));
  });

  if (!filteredEntries.length) {
    console.warn("[HistogramViewer] No histogram entries with matching bins");
    return;
  }

  const axisSpan =
    axisValues.length > 1
      ? Math.max(...axisValues) - Math.min(...axisValues)
      : 0;
  const axisUnit =
    xAxisType.value === "relative_walltime"
      ? resolveRelativeTimeUnit(axisSpan)
      : null;

  const zMatrix = Array.from({ length: binCenters.length }, () => []);
  const customDataMatrix = Array.from({ length: binCenters.length }, () => []);

  filteredEntries.forEach((entry, entryIdx) => {
    const counts = entry.counts || [];
    const formattedValue = formatAxisValue(
      axisValues[entryIdx],
      xAxisType.value,
      axisUnit,
    );

    for (let binIdx = 0; binIdx < counts.length; binIdx++) {
      const value = counts[binIdx];
      zMatrix[binIdx].push(value);
      customDataMatrix[binIdx].push(formattedValue);
    }
  });

  const columnCount = axisValues.length;
  if (columnCount === 0) return;

  const valueRange = computeGlobalValueRange(filteredEntries, binEdges);
  let normalizedZ = zMatrix;
  if (normalize.value === "per-step") {
    const numBins = binCenters.length;

    normalizedZ = Array(numBins)
      .fill(null)
      .map(() => Array(columnCount).fill(0));

    for (let colIdx = 0; colIdx < columnCount; colIdx++) {
      const columnValues = zMatrix.map((binRow) => binRow[colIdx]);
      const maxInColumn = Math.max(...columnValues);
      for (let binIdx = 0; binIdx < numBins; binIdx++) {
        normalizedZ[binIdx][colIdx] =
          maxInColumn > 0 ? zMatrix[binIdx][colIdx] / maxInColumn : 0;
      }
    }
  } else {
    const flatValues = zMatrix.flat();
    const globalMax =
      flatValues.length > 0 ? Math.max(...flatValues) : Number.NaN;
    normalizedZ = zMatrix.map((binRow) =>
      binRow.map((v) => (globalMax > 0 ? v / globalMax : 0)),
    );
  }

  const tickInfo = buildTickInfo(axisValues, xAxisType.value, axisUnit);
  const xRange = computeAxisRange(axisValues);

  const trace = {
    type: "heatmapgl",
    x: axisValues,
    y: binCenters,
    z: normalizedZ,
    customdata: customDataMatrix,
    colorscale: colorscale.value,
    showscale: false,
    hoverongaps: false,
    zsmooth: false,
    hovertemplate:
      `<b>${getAxisTitle(xAxisType.value, axisUnit)} %{customdata}</b><br>` +
      "Bin Value: %{y:.4f}<br>" +
      "Normalized Density: %{z:.3f}<br>" +
      "<extra></extra>",
  };

  const tickConfig =
    tickInfo.tickvals.length > 0
      ? {
          tickmode: "array",
          tickvals: tickInfo.tickvals,
          ticktext: tickInfo.ticktext,
        }
      : {};

  const layout = {
    xaxis: {
      gridcolor: colors.grid,
      color: colors.text,
      title: getAxisTitle(xAxisType.value, axisUnit),
      range: xRange,
      ...tickConfig,
      fixedrange: false,
    },
    yaxis: {
      gridcolor: colors.grid,
      color: colors.text,
      title: "Bin Value",
      range: [valueRange.min, valueRange.max],
      fixedrange: false,
    },
    height: props.height - 60,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 10, r: 10, b: 40, l: 70 },
    autosize: true,
    hovermode: "closest",
  };

  Plotly.react(plotDiv.value, [trace], layout, {
    responsive: true,
    displayModeBar: false,
    doubleClick: "reset", // Enable double-click to reset zoom
  }).then(() => {
    // Clear any ghost hover tooltips
    Plotly.Fx.hover(plotDiv.value, []);

    // Add manual double-click handler (heatmapgl doesn't always respect config)
    plotDiv.value.on("plotly_doubleclick", () => {
      resetZoom();
      return false; // Prevent default
    });
  });
}

function resetZoom() {
  console.log(
    "[HistogramViewer] resetZoom called, viewMode:",
    viewMode.value,
    "plotDiv:",
    !!plotDiv.value,
  );
  if (!plotDiv.value) {
    console.warn("[HistogramViewer] resetZoom aborted - no plotDiv");
    return;
  }

  if (viewMode.value === "flow") {
    console.log("[HistogramViewer] Resetting flow mode zoom");
    if (!props.histogramData || props.histogramData.length === 0) return;

    // Recalculate original tight ranges
    let rate = props.downsampleRate ?? 1;
    if (rate === -1) {
      const targetEntries = 200;
      rate =
        props.histogramData.length <= targetEntries
          ? 1
          : Math.ceil(props.histogramData.length / targetEntries);
    }

    const sampledData =
      rate > 1
        ? props.histogramData.filter((_, idx) => idx % rate === 0)
        : props.histogramData;

    const referenceEntry = sampledData.find(
      (entry) => Array.isArray(entry.bins) && entry.bins.length > 1,
    );
    if (!referenceEntry) return;

    const binEdges = referenceEntry.bins;
    const expectedBinCount = binEdges.length;

    const axisValues = [];
    const filteredEntries = [];
    sampledData.forEach((entry) => {
      if (
        Array.isArray(entry.counts) &&
        entry.counts.length === expectedBinCount - 1 &&
        Array.isArray(entry.bins) &&
        entry.bins.length === expectedBinCount
      ) {
        axisValues.push(getAxisValue(entry, xAxisType.value));
        filteredEntries.push(entry);
      }
    });

    if (!axisValues.length || !filteredEntries.length) return;

    const axisSpan =
      axisValues.length > 1
        ? Math.max(...axisValues) - Math.min(...axisValues)
        : 0;
    const axisUnit =
      xAxisType.value === "relative_walltime"
        ? resolveRelativeTimeUnit(axisSpan)
        : null;
    const tickInfo = buildTickInfo(axisValues, xAxisType.value, axisUnit);
    const xRange = computeAxisRange(axisValues);
    const valueRange = computeGlobalValueRange(filteredEntries, binEdges);

    const relayoutUpdate = {
      "xaxis.range": xRange,
      "yaxis.range": [valueRange.min, valueRange.max],
    };
    if (tickInfo.tickvals.length > 0) {
      relayoutUpdate["xaxis.tickmode"] = "array";
      relayoutUpdate["xaxis.tickvals"] = tickInfo.tickvals;
      relayoutUpdate["xaxis.ticktext"] = tickInfo.ticktext;
    } else {
      relayoutUpdate["xaxis.tickmode"] = "auto";
      relayoutUpdate["xaxis.tickvals"] = null;
      relayoutUpdate["xaxis.ticktext"] = null;
    }

    Plotly.relayout(plotDiv.value, relayoutUpdate);
    console.log(
      "[HistogramViewer] Reset flow ranges, y=[" +
        valueRange.min +
        "," +
        valueRange.max +
        "]",
    );
  } else {
    console.log("[HistogramViewer] Resetting single step mode");
    createPlot();
  }
}

defineExpose({
  resetView: resetZoom,
});
</script>

<template>
  <div
    class="histogram-viewer flex flex-col"
    :style="{ height: `${height}px` }"
  >
    <div
      v-if="histogramData && histogramData.length > 0"
      class="flex flex-col h-full"
    >
      <div class="flex-1 min-h-0">
        <div v-if="surfaceModeActive" class="h-full flex flex-col gap-2">
          <div
            class="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400"
          >
            3D Surface View
          </div>
          <div v-if="shouldShowSurface" class="flex-1 min-h-[200px]">
            <HistogramSurfaceMini
              :surface-data="props.surfaceData"
              :height="surfacePlotHeight"
              :axis-label="surfaceAxisLabel"
              value-label="Bin Value"
              :density-label="surfaceDensityLabel"
              :aspect="props.surfaceAspect"
            />
          </div>
          <div
            v-else
            class="flex-1 min-h-[200px] flex items-center justify-center rounded-lg border border-dashed border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
          >
            <el-skeleton
              v-if="props.surfaceLoading"
              :rows="3"
              animated
              class="w-full px-6"
            />
            <el-empty v-else description="Surface data unavailable" />
          </div>
        </div>
        <div v-else ref="plotDiv" class="flex-1 relative">
          <div
            v-if="isShiftPressed && viewMode === 'single'"
            class="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-xs font-bold z-10"
          >
            SYNC MODE
          </div>
        </div>
      </div>

      <!-- Slider (only for single step mode) -->
      <div v-if="viewMode === 'single'" class="mt-2 flex justify-center">
        <div class="w-1/2">
          <div
            class="text-sm text-gray-600 dark:text-gray-400 mb-2 text-center"
          >
            Step: {{ currentHistogram.step }}
            <span v-if="isShiftPressed" class="text-blue-500 font-bold ml-2">
              (Shift pressed - syncing all sliders)
            </span>
          </div>
          <el-slider
            v-model="stepIndex"
            :min="0"
            :max="histogramData.length - 1"
            :marks="{ 0: 'Start', [histogramData.length - 1]: 'End' }"
            :format-tooltip="
              (index) => `Step: ${histogramData[index]?.step ?? index}`
            "
          />
        </div>
      </div>
    </div>
    <el-empty v-else description="No histogram data" />

    <!-- Histogram settings dialog -->
    <el-dialog v-model="showSettings" title="Histogram Settings" width="520px">
      <el-form label-width="140px">
        <el-form-item label="View Mode">
          <el-select v-model="viewMode" class="w-full">
            <el-option label="Single Step" value="single" />
            <el-option label="Distribution Flow" value="flow" />
          </el-select>
        </el-form-item>

        <template v-if="viewMode === 'flow'">
          <el-divider content-position="left">Flow Settings</el-divider>
          <el-form-item label="X-Axis">
            <el-select v-model="xAxisType" class="w-full">
              <el-option label="Training Step (auto-increment)" value="step" />
              <el-option
                label="Global Step (optimizer step)"
                value="global_step"
              />
            </el-select>
          </el-form-item>
          <el-form-item label="Colorscale">
            <el-select v-model="colorscale" class="w-full">
              <el-option label="Viridis" value="Viridis" />
              <el-option label="Hot" value="Hot" />
              <el-option label="Blues" value="Blues" />
              <el-option label="RdBu" value="RdBu" />
              <el-option label="Jet" value="Jet" />
            </el-select>
          </el-form-item>
          <el-form-item label="Normalization">
            <el-select v-model="normalize" class="w-full">
              <el-option label="Per-step (recommended)" value="per-step" />
              <el-option label="Global (across all steps)" value="global" />
            </el-select>
          </el-form-item>
        </template>
      </el-form>
    </el-dialog>
  </div>
</template>
