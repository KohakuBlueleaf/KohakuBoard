<script setup>
import Plotly from "plotly.js-dist-min";
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
});

const emit = defineEmits(["update:currentStep", "update:mode", "update:xAxis"]);

const plotDiv = ref(null);
let resizeObserver = null;
const viewMode = ref(props.initialMode);
const colorscale = ref("Viridis");
const normalize = ref("per-step"); // Default per-step for better contrast
const showSettings = ref(false);
const xAxisType = ref(props.xAxis);

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

  const layout = {
    xaxis: { gridcolor: colors.grid, color: colors.text, title: "Value" },
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

  // Build heatmap data: x=steps, y=bin values, z=counts
  const steps = [];
  const allBins = sampledData[0].bins; // Use first entry's bins
  const binCenters = allBins.slice(0, -1).map((bin, i) => {
    return (bin + allBins[i + 1]) / 2;
  });

  // Z matrix: [bin_idx][step_idx] = count
  const zMatrix = Array(binCenters.length)
    .fill(null)
    .map(() => []);

  for (const entry of sampledData) {
    // Use selected x-axis (step or global_step)
    const xValue =
      xAxisType.value === "global_step" ? entry.global_step : entry.step;
    steps.push(xValue);

    // Add counts for each bin
    for (let binIdx = 0; binIdx < entry.counts.length; binIdx++) {
      zMatrix[binIdx].push(entry.counts[binIdx]);
    }
  }

  // Normalize counts if needed
  let normalizedZ = zMatrix;
  if (normalize.value === "per-step") {
    // Normalize each COLUMN (step) to 0-1
    // Need to transpose, normalize, then transpose back
    const numSteps = steps.length;
    const numBins = binCenters.length;

    normalizedZ = Array(numBins)
      .fill(null)
      .map(() => Array(numSteps).fill(0));

    for (let stepIdx = 0; stepIdx < numSteps; stepIdx++) {
      // Get all bins for this step
      const stepColumn = zMatrix.map((binRow) => binRow[stepIdx]);
      const maxInStep = Math.max(...stepColumn);

      // Normalize this step's column
      for (let binIdx = 0; binIdx < numBins; binIdx++) {
        normalizedZ[binIdx][stepIdx] =
          maxInStep > 0 ? zMatrix[binIdx][stepIdx] / maxInStep : 0;
      }
    }
  } else {
    // Global normalization
    const globalMax = Math.max(...zMatrix.flat());
    normalizedZ = zMatrix.map((binRow) =>
      binRow.map((v) => (globalMax > 0 ? v / globalMax : v)),
    );
  }

  const trace = {
    type: "heatmapgl", // Use WebGL for better performance
    x: steps,
    y: binCenters,
    z: normalizedZ,
    colorscale: colorscale.value,
    showscale: false, // Hide color bar
    hoverongaps: false,
    zsmooth: false, // Disable smoothing (steps may not be evenly spaced)
    hovertemplate:
      "<b>Step %{x}</b><br>" +
      "Bin Value: %{y:.4f}<br>" +
      "Normalized Density: %{z:.3f}<br>" +
      "<extra></extra>",
  };

  // Calculate exact axis ranges to eliminate padding
  const xMin = Math.min(...steps);
  const xMax = Math.max(...steps);
  const yMin = Math.min(...binCenters);
  const yMax = Math.max(...binCenters);

  const layout = {
    xaxis: {
      gridcolor: colors.grid,
      color: colors.text,
      title: "Training Step",
      range: [xMin, xMax], // Exact range, no padding
      fixedrange: false,
    },
    yaxis: {
      gridcolor: colors.grid,
      color: colors.text,
      title: "Bin Value",
      range: [yMin, yMax], // Exact range, no padding
      fixedrange: false,
    },
    height: props.height - 60,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 10, r: 10, b: 40, l: 70 }, // More left margin for y-axis labels
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
    const steps = sampledData.map((entry) =>
      xAxisType.value === "global_step" ? entry.global_step : entry.step,
    );
    const allBins = sampledData[0].bins;
    const binCenters = allBins
      .slice(0, -1)
      .map((bin, i) => (bin + allBins[i + 1]) / 2);

    const xMin = Math.min(...steps);
    const xMax = Math.max(...steps);
    const yMin = Math.min(...binCenters);
    const yMax = Math.max(...binCenters);

    // Only reset axes, keep height unchanged
    Plotly.relayout(plotDiv.value, {
      "xaxis.range": [xMin, xMax],
      "yaxis.range": [yMin, yMax],
    });
    console.log(
      "[HistogramViewer] Reset to ranges: x=[" +
        xMin +
        "," +
        xMax +
        "], y=[" +
        yMin +
        "," +
        yMax +
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
      <div ref="plotDiv" class="flex-1 relative">
        <!-- Shift indicator -->
        <div
          v-if="isShiftPressed && viewMode === 'single'"
          class="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-xs font-bold z-10"
        >
          SYNC MODE
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
    <el-dialog v-model="showSettings" title="Histogram Settings" width="400px">
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
