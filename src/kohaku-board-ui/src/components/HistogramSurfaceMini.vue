<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const DEFAULT_RELATIVE_UNIT = {
  scale: 1,
  label: "Seconds",
  suffix: " s",
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

const props = defineProps({
  surfaceData: {
    type: Object,
    default: null,
  },
  height: {
    type: Number,
    default: 380,
  },
  axisLabel: {
    type: String,
    default: "Training Step",
  },
  valueLabel: {
    type: String,
    default: "Bin Value",
  },
  densityLabel: {
    type: String,
    default: "Normalized Density",
  },
  aspect: {
    type: Object,
    default: () => ({
      x: 1.1,
      y: 1,
      z: 0.75,
    }),
  },
});

const DEFAULT_MINI_ASPECT = { x: 1.1, y: 1, z: 0.75 };

const plotDiv = ref(null);
const plotContainer = ref(null);
let resizeObserver = null;
let relayoutHandler = null;
let cameraState = null;

const hasData = computed(() => {
  const payload = props.surfaceData;
  return (
    payload &&
    Array.isArray(payload.matrix) &&
    payload.matrix.length > 0 &&
    Array.isArray(payload.bin_centers) &&
    payload.bin_centers.length > 0 &&
    Array.isArray(payload.axis_values) &&
    payload.axis_values.length > 0
  );
});

watch(
  () => props.surfaceData,
  () => {
    nextTick(() => createPlot());
  },
  { deep: true },
);

watch(
  () => props.height,
  () => {
    nextTick(() => createPlot());
  },
);

watch(
  () => props.aspect,
  () => {
    nextTick(() => createPlot());
  },
  { deep: true },
);

const aspectRatio = computed(() => {
  const aspect = props.aspect || DEFAULT_MINI_ASPECT;
  return {
    x: Number.isFinite(aspect.x) ? aspect.x : DEFAULT_MINI_ASPECT.x,
    y: Number.isFinite(aspect.y) ? aspect.y : DEFAULT_MINI_ASPECT.y,
    z: Number.isFinite(aspect.z) ? aspect.z : DEFAULT_MINI_ASPECT.z,
  };
});

onMounted(() => {
  createPlot();
  setupResizeObserver();
  watchThemeChanges();
});

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect();
  }
  if (plotDiv.value && relayoutHandler) {
    plotDiv.value.removeListener?.("plotly_relayout", relayoutHandler);
  }
});

function isDarkMode() {
  return document.documentElement.classList.contains("dark");
}

function getThemeColors() {
  const dark = isDarkMode();
  return {
    text: dark ? "#e5e7eb" : "#1f2937",
    grid: dark ? "rgba(156, 163, 175, 0.2)" : "rgba(156, 163, 175, 0.3)",
    bg: dark ? "#111827" : "#ffffff",
  };
}

function createPlot() {
  if (!plotDiv.value || !hasData.value) {
    if (plotDiv.value) {
      Plotly.purge(plotDiv.value);
    }
    return;
  }

  const payload = props.surfaceData;
  const colors = getThemeColors();
  const axisInfo = buildAxisInfo(payload);
  const matrix = payload.matrix || [];
  const normalizeMode = payload?.normalize || "none";
  const matrixMax = computeMatrixMax(matrix);
  const containerHeight =
    plotContainer.value?.clientHeight ?? Math.max(props.height, 240);
  const rawMax = Number.isFinite(payload.raw_max_value)
    ? payload.raw_max_value
    : matrixMax;
  const densityRange = resolveDensityRange(normalizeMode, matrixMax, rawMax);
  const trace = {
    type: "surface",
    x: payload.bin_centers,
    y: axisInfo.values,
    z: matrix,
    colorscale: "Turbo",
    showscale: true,
    hovertemplate:
      `${props.valueLabel}: %{x:.4f}<br>` +
      `${axisInfo.label}: %{y:.4f}${axisInfo.suffix}<br>` +
      `${props.densityLabel}: %{z:.4f}<extra></extra>`,
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    height: containerHeight,
    margin: { t: 10, b: 0, l: 0, r: 0 },
    font: { color: colors.text },
    scene: {
      bgcolor: colors.bg,
      aspectmode: "manual",
      aspectratio: aspectRatio.value,
      xaxis: {
        title: props.valueLabel,
        color: colors.text,
        gridcolor: colors.grid,
        zerolinecolor: colors.grid,
      },
      yaxis: {
        title: axisInfo.label,
        color: colors.text,
        gridcolor: colors.grid,
        zerolinecolor: colors.grid,
      },
      zaxis: {
        title: props.densityLabel,
        color: colors.text,
        gridcolor: colors.grid,
        zerolinecolor: colors.grid,
        range: [densityRange.min, Math.max(densityRange.max, 0.1)],
      },
      camera: cameraState || {
        eye: { x: 1.6, y: -1.6, z: 0.9 },
      },
    },
  };

  Plotly.react(plotDiv.value, [trace], layout, {
    responsive: true,
    displayModeBar: false,
  }).then(() => {
    bindRelayoutHandler();
  });
}

function computeMatrixMax(matrix) {
  if (!Array.isArray(matrix)) return 0;
  let maxValue = Number.NEGATIVE_INFINITY;
  for (const row of matrix) {
    if (!Array.isArray(row)) continue;
    for (const value of row) {
      if (Number.isFinite(value) && value > maxValue) {
        maxValue = value;
      }
    }
  }
  return Number.isFinite(maxValue) ? maxValue : 0;
}

function buildAxisInfo(payload) {
  const axisType = payload?.axis_type || payload?.axis_requested;
  const axisValues = Array.isArray(payload?.axis_values)
    ? payload.axis_values
    : [];
  const stats = payload?.axis_stats || {};
  if (axisType === "relative_walltime") {
    const span =
      Number.isFinite(stats.max) && Number.isFinite(stats.min)
        ? stats.max - stats.min
        : 0;
    const unit = resolveRelativeTimeUnit(span);
    return {
      values: axisValues.map((val) =>
        Number.isFinite(val) ? val / unit.scale : val,
      ),
      label: `Time (${unit.label})`,
      suffix: unit.suffix,
    };
  }
  return {
    values: axisValues,
    label: payload?.axis_label || "Step",
    suffix: "",
  };
}

function resolveDensityRange(normalizeMode, matrixMax, rawMax) {
  const normalized = normalizeMode && normalizeMode !== "none";
  const matrixUpper = Number.isFinite(matrixMax) ? matrixMax : 0;
  const rawUpper = Number.isFinite(rawMax) ? rawMax : matrixUpper;

  if (normalized) {
    const upper = Math.max(matrixUpper, 1);
    const padding = Math.max(upper * 0.05, 0.05);
    return { min: 0, max: upper + padding };
  }

  const upper = rawUpper > 0 ? rawUpper : 1;
  const padding = Math.max(upper * 0.05, 0.1);
  return { min: 0, max: upper + padding };
}

function bindRelayoutHandler() {
  if (!plotDiv.value || !plotDiv.value.on) return;
  if (relayoutHandler) {
    plotDiv.value.removeListener("plotly_relayout", relayoutHandler);
  }
  relayoutHandler = (eventData) => {
    if (eventData["scene.camera"]) {
      cameraState = eventData["scene.camera"];
    }
  };
  plotDiv.value.on("plotly_relayout", relayoutHandler);
}

function resetCamera() {
  cameraState = {
    eye: { x: 1.6, y: -1.6, z: 0.9 },
  };
  if (plotDiv.value) {
    Plotly.relayout(plotDiv.value, { "scene.camera": cameraState });
  }
}

function setupResizeObserver() {
  if (resizeObserver) {
    resizeObserver.disconnect();
  }
  const target = plotContainer.value || plotDiv.value;
  if (!target) return;
  resizeObserver = new ResizeObserver(() => {
    if (plotDiv.value) {
      Plotly.Plots.resize(plotDiv.value);
    }
  });
  resizeObserver.observe(target);
}

function watchThemeChanges() {
  const observer = new MutationObserver(() => {
    createPlot();
  });
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  onUnmounted(() => observer.disconnect());
}

defineExpose({
  resetCamera,
});
</script>

<template>
  <div
    ref="plotContainer"
    class="w-full h-full"
    :style="{ minHeight: `${height}px` }"
  >
    <div
      v-if="hasData"
      ref="plotDiv"
      class="w-full h-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900"
    />
    <el-empty
      v-else
      description="Surface data unavailable"
      :image-size="120"
      class="bg-white dark:bg-gray-900 rounded-lg border border-dashed border-gray-300 dark:border-gray-700 py-8 h-full flex items-center justify-center"
    />
  </div>
</template>
