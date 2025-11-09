<script setup>
import { ref, reactive, onMounted, watch, onUnmounted, computed } from "vue";
import Plotly from "plotly.js-dist-min";
import { useAnimationPreference } from "@/composables/useAnimationPreference";
import { useHoverSync } from "@/composables/useHoverSync";

const { animationsEnabled } = useAnimationPreference();
const { registerChart, unregisterChart } = useHoverSync();

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

function scaleRelativeValue(seconds, unit = DEFAULT_RELATIVE_UNIT) {
  if (!Number.isFinite(seconds)) return seconds;
  return seconds / unit.scale;
}

function computeRelativeAxisInfo(seriesList) {
  let min = Infinity;
  let max = -Infinity;

  seriesList.forEach((series) => {
    series.segments?.forEach((seg) => {
      seg.x?.forEach((xVal) => {
        if (Number.isFinite(xVal)) {
          if (xVal < min) min = xVal;
          if (xVal > max) max = xVal;
        }
      });
    });
  });

  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return { ...DEFAULT_RELATIVE_UNIT };
  }

  return resolveRelativeTimeUnit(max - min);
}

// No emits needed - parent watches plotConfig directly via ref

const props = defineProps({
  data: {
    type: Array,
    required: true,
    // Format: [{ name: 'metric1', x: [...], y: [...] }, ...]
  },
  title: {
    type: String,
    default: "",
  },
  xaxis: {
    type: String,
    default: "Step",
  },
  yaxis: {
    type: String,
    default: "Value",
  },
  height: {
    type: Number,
    default: 400,
  },
  hideToolbar: {
    type: Boolean,
    default: false,
  },
  // Config props (can be controlled by parent)
  smoothingMode: {
    type: String,
    default: undefined,
  },
  smoothingValue: {
    type: Number,
    default: undefined,
  },
  downsampleRate: {
    type: Number,
    default: undefined,
  },
  showOriginal: {
    type: Boolean,
    default: undefined,
  },
  lineWidth: {
    type: Number,
    default: undefined,
  },
  showMarkers: {
    type: Boolean,
    default: undefined,
  },
  xRange: {
    type: Object,
    default: undefined,
  },
  yRange: {
    type: Object,
    default: undefined,
  },
  // Hover sync props
  tabName: {
    type: String,
    default: "",
  },
  chartId: {
    type: String,
    required: true,
  },
  hoverSyncEnabled: {
    type: Boolean,
    default: true,
  },
  // Multi-run props
  multiRunMode: {
    type: Boolean,
    default: false,
  },
  runColors: {
    type: Object,
    default: () => ({}),
  },
  runNames: {
    type: Object,
    default: () => ({}),
  },
});

const plotDiv = ref(null);
let myResizeObserver = null;
let unregisterHoverSync = null;
const showConfigModal = ref(false);
const modalMouseDownTarget = ref(null);

// Internal config (can be overridden by props)
const config = reactive({
  xRange: props.xRange ?? { auto: true, min: null, max: null },
  yRange: props.yRange ?? { auto: true, min: null, max: null },
  smoothingMode: props.smoothingMode ?? "disabled",
  smoothingValue: props.smoothingValue ?? 0.9,
  downsampleRate: props.downsampleRate ?? -1, // -1 = adaptive
  showOriginal: props.showOriginal ?? true,
  showMarkers: props.showMarkers ?? false,
  lineWidth: props.lineWidth ?? 1.5,
});

// Watch for prop changes and update internal config
watch(
  () => props.smoothingMode,
  (val) => {
    if (val !== undefined) config.smoothingMode = val;
  },
);
watch(
  () => props.smoothingValue,
  (val) => {
    if (val !== undefined) config.smoothingValue = val;
  },
);
watch(
  () => props.downsampleRate,
  (val) => {
    if (val !== undefined) config.downsampleRate = val;
  },
);
watch(
  () => props.showOriginal,
  (val) => {
    if (val !== undefined) config.showOriginal = val;
  },
);
watch(
  () => props.lineWidth,
  (val) => {
    if (val !== undefined) config.lineWidth = val;
  },
);
watch(
  () => props.showMarkers,
  (val) => {
    if (val !== undefined) config.showMarkers = val;
  },
);
watch(
  () => props.xRange,
  (val) => {
    if (val !== undefined) config.xRange = val;
  },
  { deep: true },
);
watch(
  () => props.yRange,
  (val) => {
    if (val !== undefined) config.yRange = val;
  },
  { deep: true },
);

onMounted(() => {
  console.log("[LinePlot] Mounted");
  createPlot();
  setupResizeObserver();
  watchThemeChanges();

  // Use nextTick to ensure plotDiv is fully initialized after createPlot
  nextTick(() => {
    setupHoverSync();
  });
});

onUpdated(() => {
  console.log("[LinePlot] Updated/re-rendered");
});

onUnmounted(() => {
  console.log("[LinePlot] Unmounting, disconnecting ResizeObserver");
  if (myResizeObserver) {
    myResizeObserver.disconnect();
  }
  if (unregisterHoverSync) {
    unregisterHoverSync();
  }
});

watch(
  () => props.data,
  () => {
    createPlot();
  },
  { deep: true },
);

watch(
  () => props.height,
  (newH, oldH) => {
    console.log(
      "[LinePlot] Height prop changed:",
      oldH,
      "→",
      newH,
      "- triggering full redraw",
    );
    createPlot();
  },
);

// Recreate plot when config changes (parent saves via ref watcher)
watch(config, () => createPlot(), { deep: true });

function applySmoothing(y, mode, value) {
  if (mode === "disabled") return y;

  const result = [];

  switch (mode) {
    case "ema": {
      let ema = y[0];
      const k = (1 - value) / 1;
      for (let i = 0; i < y.length; i++) {
        const decay = Math.min(i / k, value);
        ema = (1 - decay) * y[i] + decay * ema;
        result.push(ema);
      }
      break;
    }
    case "gaussian": {
      const kernelSize = Math.max(3, Math.floor(value));
      const sigma = kernelSize / 6.0;
      const kernel = [];
      const halfSize = Math.floor(kernelSize / 2);

      let sum = 0;
      for (let i = -halfSize; i <= halfSize; i++) {
        const val = Math.exp(-(i * i) / (2 * sigma * sigma));
        kernel.push(val);
        sum += val;
      }
      for (let i = 0; i < kernel.length; i++) {
        kernel[i] /= sum;
      }

      for (let i = 0; i < y.length; i++) {
        let weighted = 0;
        let weightSum = 0;
        for (let j = -halfSize; j <= halfSize; j++) {
          const idx = i + j;
          if (idx >= 0 && idx < y.length) {
            weighted += y[idx] * kernel[j + halfSize];
            weightSum += kernel[j + halfSize];
          }
        }
        result.push(weighted / weightSum);
      }
      break;
    }
    case "ma": {
      const window = Math.max(1, Math.floor(value));
      for (let i = 0; i < y.length; i++) {
        const start = Math.max(0, i - window + 1);
        let sum = 0;
        for (let j = start; j <= i; j++) {
          sum += y[j];
        }
        result.push(sum / (i - start + 1));
      }
      break;
    }
    default:
      return y;
  }

  return result;
}

function downsampleData(x, y, rate) {
  // Adaptive downsampling if rate is -1
  if (rate === -1) {
    // Target: ~5000 points for smooth rendering
    const targetPoints = 5000;
    if (x.length <= targetPoints) {
      return { x, y }; // No downsampling needed
    }
    // Calculate adaptive rate
    rate = Math.ceil(x.length / targetPoints);
  }

  if (rate <= 1) return { x, y };

  const newX = [];
  const newY = [];
  for (let i = 0; i < x.length; i++) {
    if (i % rate === 0) {
      newX.push(x[i]);
      newY.push(y[i]);
    }
  }
  return { x: newX, y: newY };
}

const processedData = computed(() => {
  const mode = config.smoothingMode;
  const value = config.smoothingValue;
  const rate = config.downsampleRate;

  // Track duplicate names and add suffixes
  const nameCount = new Map();

  return props.data.map((series) => {
    // In multi-run mode, replace run_id with run_name in series name
    let displayName = series.name;
    let originalRunId = null; // Keep track of original run_id for color lookup

    if (props.multiRunMode && series.name.includes(" (")) {
      const runIdMatch = series.name.match(/\(([^)]+)\)$/);
      if (runIdMatch) {
        const runId = runIdMatch[1];
        originalRunId = runId; // Store for color lookup
        const runName = props.runNames[runId] || runId;
        const baseMetric = series.name.substring(
          0,
          series.name.lastIndexOf(" ("),
        );
        displayName = `${baseMetric} (${runName})`;

        // Handle duplicate names
        if (nameCount.has(displayName)) {
          const count = nameCount.get(displayName) + 1;
          nameCount.set(displayName, count);
          displayName = `${baseMetric} (${runName}_${count})`;
        } else {
          nameCount.set(displayName, 1);
        }
      }
    }

    console.log(
      `[LinePlot] Processing series: ${series.name} -> ${displayName}, data length: ${series.y.length}`,
    );

    // Build line segments separated by NaN/inf breaks
    // If we have: [1, 2, NaN, 3, 4], we create 2 segments: [[1,2], [3,4]]
    const segments = []; // Each segment: {x: [], y: []}
    const nanX = [];
    const infX = [];
    const negInfX = [];

    let currentSegmentX = [];
    let currentSegmentY = [];

    for (let i = 0; i < series.y.length; i++) {
      const yVal = series.y[i];
      const xVal = series.x[i];

      if (Number.isNaN(yVal)) {
        // NaN breaks the line - finish current segment
        if (currentSegmentX.length > 0) {
          segments.push({ x: currentSegmentX, y: currentSegmentY });
          currentSegmentX = [];
          currentSegmentY = [];
        }
        nanX.push(xVal);
        console.log(
          `[LinePlot] ${series.name}[${i}]: NaN at x=${xVal} - breaking line`,
        );
      } else if (yVal === Infinity) {
        // +inf breaks the line
        if (currentSegmentX.length > 0) {
          segments.push({ x: currentSegmentX, y: currentSegmentY });
          currentSegmentX = [];
          currentSegmentY = [];
        }
        infX.push(xVal);
        console.log(
          `[LinePlot] ${series.name}[${i}]: +inf at x=${xVal} - breaking line`,
        );
      } else if (yVal === -Infinity) {
        // -inf breaks the line
        if (currentSegmentX.length > 0) {
          segments.push({ x: currentSegmentX, y: currentSegmentY });
          currentSegmentX = [];
          currentSegmentY = [];
        }
        negInfX.push(xVal);
        console.log(
          `[LinePlot] ${series.name}[${i}]: -inf at x=${xVal} - breaking line`,
        );
      } else {
        // Normal finite value - add to current segment
        currentSegmentX.push(xVal);
        currentSegmentY.push(yVal);
      }
    }

    // Don't forget the last segment
    if (currentSegmentX.length > 0) {
      segments.push({ x: currentSegmentX, y: currentSegmentY });
    }

    console.log(
      `[LinePlot] ${series.name}: ${segments.length} segments, NaN=${nanX.length}, +inf=${infX.length}, -inf=${negInfX.length}`,
    );

    // Process each segment separately (smoothing + downsampling)
    const processedSegments = segments.map((seg) => {
      const smoothedY = applySmoothing(seg.y, mode, value);
      return downsampleData(seg.x, smoothedY, rate);
    });

    const result = {
      name: displayName, // Use display name (with run_name instead of run_id)
      segments: processedSegments, // Multiple line segments
      original: null,
      originalSegments: null,
      // Store special value x-positions for marker rendering
      nanX,
      infX,
      negInfX,
      // Store original run_id for color lookup
      runId: originalRunId,
    };

    if (mode !== "disabled" && config.showOriginal) {
      const originalSegments = segments.map((seg) =>
        downsampleData(seg.x, seg.y, rate),
      );
      result.originalSegments = originalSegments;
    }

    return result;
  });
});

function isDarkMode() {
  return document.documentElement.classList.contains("dark");
}

function getThemeColors() {
  const dark = isDarkMode();
  return {
    text: dark ? "#e5e7eb" : "#1f2937",
    grid: dark ? "rgba(156, 163, 175, 0.2)" : "rgba(156, 163, 175, 0.3)",
    zeroline: dark ? "rgba(156, 163, 175, 0.3)" : "rgba(156, 163, 175, 0.4)",
  };
}

function createPlot() {
  if (!plotDiv.value) return;

  const colors = getThemeColors();
  const data = processedData.value;
  const relativeAxisInfo =
    props.xaxis === "relative_walltime" ? computeRelativeAxisInfo(data) : null;
  const scaleXValues = (arr) =>
    !relativeAxisInfo || relativeAxisInfo.scale === 1
      ? arr
      : arr.map((value) => scaleRelativeValue(value, relativeAxisInfo));
  const formatRelativeValueForHover = (seconds) =>
    formatRelativeDurationValue(
      seconds,
      relativeAxisInfo || DEFAULT_RELATIVE_UNIT,
    );

  const plotlyColors = [
    "#3b82f6",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ];

  // Helper to convert hex to RGBA
  const hexToRgba = (hex, alpha) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Collect traces in two groups: original first, then smoothed (so smoothed renders on top)
  const originalTraces = [];
  const smoothedTraces = [];
  const specialMarkerTraces = [];

  data.forEach((series, index) => {
    console.log(
      `[LinePlot] createPlot - processing series ${series.name}: ${series.segments?.length || 0} segments, nanX=${series.nanX?.length || 0}, infX=${series.infX?.length || 0}, negInfX=${series.negInfX?.length || 0}`,
    );

    // In multi-run mode, use the stored run_id to look up color
    let baseColor = plotlyColors[index % plotlyColors.length];
    if (props.multiRunMode && series.runId) {
      const runId = series.runId;
      if (props.runColors[runId]) {
        baseColor = props.runColors[runId];
        console.log(
          `[LinePlot] Using run color for ${series.name}: ${baseColor} (runId: ${runId})`,
        );
      } else {
        console.warn(
          `[LinePlot] No color found for runId: ${runId}, available:`,
          Object.keys(props.runColors),
        );
      }
    }

    // Store original data for hover template
    const hasSmoothing = config.smoothingMode !== "disabled";
    const originalSegments = series.originalSegments || [];

    // Render original segments if enabled (pushed first, renders below)
    if (originalSegments.length > 0 && config.showOriginal) {
      // Dimmer color for original data
      const dimmerColor = hexToRgba(baseColor, 0.3); // Lower opacity for original

      originalSegments.forEach((seg, segIdx) => {
        originalTraces.push({
          type: "scattergl",
          mode: "lines",
          name: series.name, // Same name, no "(original)" suffix
          x: scaleXValues(seg.x),
          y: seg.y,
          line: {
            width: config.lineWidth * 0.5,
            color: dimmerColor,
          },
          showlegend: false, // Hide from legend
          legendgroup: series.name,
          hoverinfo: "skip", // Don't show in hover
        });
      });
    }

    // Render smoothed/main line segments (pushed last, renders on top)
    if (series.segments && series.segments.length > 0) {
      series.segments.forEach((seg, segIdx) => {
        const useTransparency = processedData.length > 1;
        const lineOpacity = useTransparency ? 0.6 : 1.0;
        const rawSegmentX = seg.x || [];
        const scaledSegmentX = scaleXValues(rawSegmentX);

        const trace = {
          type: "scattergl",
          mode: config.showMarkers ? "lines+markers" : "lines",
          name: series.name,
          x: scaledSegmentX,
          y: seg.y,
          line: {
            width: config.lineWidth,
            color: hexToRgba(baseColor, lineOpacity),
          },
          marker: {
            size: 3,
            color: baseColor,
            opacity: 1,
          },
          showlegend: segIdx === 0, // Only first segment in legend
          legendgroup: series.name,
        };

        // Custom hover template showing smoothed(original) if smoothing enabled
        if (hasSmoothing && originalSegments.length > 0) {
          // Build customdata with original values for hover
          const originalHoverData = rawSegmentX.map((xVal) => {
            // Find original y value at this x position
            let originalY = null;
            for (const origSeg of originalSegments) {
              const origIdx = origSeg.x.indexOf(xVal);
              if (origIdx !== -1) {
                originalY = origSeg.y[origIdx];
                break;
              }
            }
            return { originalY };
          });
          trace.customdata = originalHoverData;

          if (props.xaxis === "relative_walltime") {
            trace.customdata = rawSegmentX.map((xVal, idx) => ({
              duration: formatRelativeValueForHover(xVal),
              originalY: originalHoverData[idx]?.originalY,
            }));
            trace.hovertemplate =
              "<b>%{fullData.name}</b><br>" +
              "Time: %{customdata.duration}<br>" +
              "Value: %{y:.4f} (%{customdata.originalY:.4f})<extra></extra>";
          } else {
            trace.hovertemplate =
              "<b>%{fullData.name}</b>: %{y:.4f} (%{customdata.originalY:.4f})<extra></extra>";
          }
        } else {
          // No smoothing - simple hover
          if (props.xaxis === "relative_walltime") {
            trace.customdata = rawSegmentX.map((xVal) => ({
              duration: formatRelativeValueForHover(xVal),
            }));
            trace.hovertemplate =
              "<b>%{fullData.name}</b><br>" +
              "Time: %{customdata.duration}<br>" +
              "Value: %{y:.4f}<extra></extra>";
          } else {
            trace.hovertemplate =
              "<b>%{fullData.name}</b>: %{y:.4f}<extra></extra>";
          }
        }

        smoothedTraces.push(trace);
      });
    }

    // Add special marker traces for NaN/inf values
    // These render as dots at top/bottom without affecting axis range
    const hasSpecialValues =
      series.nanX.length > 0 ||
      series.infX.length > 0 ||
      series.negInfX.length > 0;

    console.log(
      `[LinePlot] ${series.name} special values: NaN=${series.nanX.length}, +inf=${series.infX.length}, -inf=${series.negInfX.length}`,
    );

    if (hasSpecialValues) {
      // NaN markers (circle at top of chart)
      if (series.nanX.length > 0) {
        console.log(
          `[LinePlot] Adding NaN markers for ${series.name}:`,
          series.nanX,
        );
        specialMarkerTraces.push({
          type: "scattergl",
          mode: "markers",
          name: `${series.name} (NaN)`,
          x: scaleXValues(series.nanX),
          y: series.nanX.map(() => 1),
          marker: {
            symbol: "circle",
            size: 8,
            color: baseColor,
            line: { color: "white", width: 1.5 },
          },
          yaxis: "y2",
          showlegend: false,
          legendgroup: series.name,
          hoverinfo: "skip",
          text: series.nanX.map(() => `${series.name}: NaN`),
        });
      }

      // +Infinity markers (triangle-up at top of chart)
      if (series.infX.length > 0) {
        console.log(
          `[LinePlot] Adding +inf markers for ${series.name}:`,
          series.infX,
        );
        specialMarkerTraces.push({
          type: "scattergl",
          mode: "markers",
          name: `${series.name} (+inf)`,
          x: scaleXValues(series.infX),
          y: series.infX.map(() => 0.95),
          marker: {
            symbol: "triangle-up",
            size: 10,
            color: baseColor,
            line: { color: "white", width: 1.5 },
          },
          yaxis: "y2",
          showlegend: false,
          legendgroup: series.name,
          hoverinfo: "skip",
          text: series.infX.map(() => `${series.name}: +inf`),
        });
      }

      // -Infinity markers (triangle-down at bottom of chart)
      if (series.negInfX.length > 0) {
        console.log(
          `[LinePlot] Adding -inf markers for ${series.name}:`,
          series.negInfX,
        );
        specialMarkerTraces.push({
          type: "scattergl",
          mode: "markers",
          name: `${series.name} (-inf)`,
          x: scaleXValues(series.negInfX),
          y: series.negInfX.map(() => 0.05),
          marker: {
            symbol: "triangle-down",
            size: 10,
            color: baseColor,
            line: { color: "white", width: 1.5 },
          },
          yaxis: "y2",
          showlegend: false,
          legendgroup: series.name,
          hoverinfo: "skip",
          text: series.negInfX.map(() => `${series.name}: -inf`),
        });
      }
    }
  });

  // Combine traces in order: original (bottom), smoothed (top), special markers (overlay)
  const traces = [...originalTraces, ...smoothedTraces, ...specialMarkerTraces];

  const xAxisConfig = config.xRange.auto
    ? { autorange: true }
    : {
        range:
          props.xaxis === "relative_walltime" && relativeAxisInfo
            ? [
                scaleRelativeValue(config.xRange.min, relativeAxisInfo),
                scaleRelativeValue(config.xRange.max, relativeAxisInfo),
              ]
            : [config.xRange.min, config.xRange.max],
      };

  const yAxisConfig = config.yRange.auto
    ? { autorange: true }
    : { range: [config.yRange.min, config.yRange.max] };

  const layout = {
    xaxis: {
      gridcolor: colors.grid,
      zerolinecolor: colors.zeroline,
      color: colors.text,
      showspikes: true,
      spikemode: "across",
      spikesnap: "cursor",
      spikecolor: isDarkMode() ? "#888888" : "#666666",
      spikethickness: 1.5,
      spikedash: "solid",
      showline: false,
      ...xAxisConfig,
    },
    yaxis: {
      gridcolor: colors.grid,
      zerolinecolor: colors.zeroline,
      color: colors.text,
      showspikes: true,
      spikemode: "across",
      spikesnap: "cursor",
      spikecolor: isDarkMode() ? "#888888" : "#666666",
      spikethickness: 1.5,
      spikedash: "solid",
      showline: false,
      ...yAxisConfig,
    },
    // Secondary y-axis for NaN/inf markers (fixed [0,1] range, overlaid)
    yaxis2: {
      overlaying: "y",
      side: "right",
      showgrid: false,
      showticklabels: false,
      showline: false,
      range: [0, 1],
      fixedrange: true,
    },
    dragmode: "zoom",
    height: props.height,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: colors.text },
    margin: { t: 10, r: 20, b: 30, l: 50 },
    hovermode: "x unified",
    hoverdistance: -1,
    spikedistance: -1,
    hoverlabel: {
      namelength: -1,
      bgcolor: isDarkMode()
        ? "rgba(31, 41, 55, 0.95)"
        : "rgba(255, 255, 255, 0.95)",
      bordercolor: colors.grid,
      font: { color: colors.text },
      align: "left",
    },
    showlegend: true,
    legend: {
      orientation: "h",
      y: -0.15,
      font: { color: colors.text },
    },
    modebar: {
      bgcolor: "transparent",
    },
    // Preserve user's zoom/pan state across data updates (prevents range flickering)
    uirevision: props.chartId,
  };

  // Format x-axis based on metric type
  if (props.xaxis.includes("step") || props.xaxis.includes("Step")) {
    layout.xaxis.hoverformat = "d"; // Integer format
  } else if (props.xaxis === "timestamp") {
    layout.xaxis.type = "date";
    layout.xaxis.hoverformat = "%Y-%m-%d %H:%M:%S"; // DateTime format
  } else if (props.xaxis === "walltime") {
    layout.xaxis.type = "date";
    layout.xaxis.hoverformat = "%Y-%m-%d %H:%M:%S"; // DateTime format
  } else if (props.xaxis === "relative_walltime") {
    const unit = relativeAxisInfo || DEFAULT_RELATIVE_UNIT;
    layout.xaxis.title = `Time (${unit.label})`;
    layout.xaxis.tickmode = "auto";
    layout.xaxis.tickformat = ".2f";
    layout.xaxis.ticksuffix = unit.suffix;
  } else {
    layout.xaxis.hoverformat = ".4f"; // Default float format
  }

  const plotConfig = {
    responsive: true,
    displayModeBar: false,
    displaylogo: false,
    plotGlPixelRatio: 2,
  };

  const transitionConfig = {
    duration: 0,
    easing: "linear",
  };

  Plotly.react(plotDiv.value, traces, layout, {
    ...plotConfig,
    transition: transitionConfig,
  });
}

function resetView() {
  config.xRange.auto = true;
  config.yRange.auto = true;
}

function exportPNG() {
  Plotly.downloadImage(plotDiv.value, {
    format: "png",
    filename: props.title || "plot",
    height: props.height,
    width: 1200,
    scale: 2,
  });
}

function openConfig() {
  showConfigModal.value = true;
}

function handleModalMouseDown(e) {
  if (e.target === e.currentTarget) {
    modalMouseDownTarget.value = e.target;
  } else {
    modalMouseDownTarget.value = null;
  }
}

function handleModalMouseUp(e) {
  if (e.target === e.currentTarget && modalMouseDownTarget.value === e.target) {
    showConfigModal.value = false;
  }
  modalMouseDownTarget.value = null;
}

function setupResizeObserver() {
  if (!plotDiv.value) return;

  myResizeObserver = new ResizeObserver((entries) => {
    console.log("[LinePlot] ResizeObserver triggered");
    if (plotDiv.value) {
      Plotly.Plots.resize(plotDiv.value);
    }
  });

  myResizeObserver.observe(plotDiv.value.parentElement || plotDiv.value);
  console.log(
    "[LinePlot] ResizeObserver setup on",
    plotDiv.value.parentElement || plotDiv.value,
  );
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

function setupHoverSync() {
  console.log(`[LinePlot] setupHoverSync called for chart ${props.chartId}`);
  console.log(`[LinePlot] Props:`, {
    hoverSyncEnabled: props.hoverSyncEnabled,
    tabName: props.tabName,
    chartId: props.chartId,
    xaxis: props.xaxis,
    hasPlotDiv: !!plotDiv.value,
  });

  if (!props.hoverSyncEnabled) {
    console.log(
      `[LinePlot] Hover sync disabled via prop, skipping registration`,
    );
    return;
  }

  if (!props.tabName) {
    console.log(`[LinePlot] No tabName provided, skipping registration`);
    return;
  }

  if (!plotDiv.value) {
    console.log(`[LinePlot] plotDiv not available yet, skipping registration`);
    return;
  }

  // Function to get current x-axis type from props
  const getXAxisType = () => {
    console.log(`[LinePlot] getXAxisType called, returning:`, props.xaxis);
    return props.xaxis;
  };

  // Register this chart for hover synchronization
  console.log(`[LinePlot] Calling registerChart for ${props.chartId}`);
  unregisterHoverSync = registerChart(
    props.tabName,
    props.chartId,
    plotDiv.value,
    getXAxisType,
  );

  console.log(
    `[LinePlot] Successfully registered for hover sync: ${props.chartId} in tab ${props.tabName}`,
  );
}

defineExpose({
  resetView,
  exportPNG,
  openConfig,
  plotConfig: config,
});
</script>

<template>
  <div class="relative line-plot-container">
    <div
      v-if="!hideToolbar"
      class="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-2 z-10"
    >
      <div class="flex-1"></div>
      <div class="flex gap-2">
        <button
          @click="resetView"
          class="p-1.5 rounded bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          title="Reset View"
        >
          <svg
            class="w-4 h-4 text-gray-700 dark:text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        </button>
        <button
          @click="exportPNG"
          class="p-1.5 rounded bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          title="Export PNG"
        >
          <svg
            class="w-4 h-4 text-gray-700 dark:text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
        </button>
      </div>
      <div class="flex-1 flex justify-end">
        <button
          @click="openConfig"
          class="p-1.5 rounded bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          title="Settings"
        >
          <svg
            class="w-4 h-4 text-gray-700 dark:text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </button>
      </div>
    </div>

    <div ref="plotDiv" class="w-full"></div>

    <div
      v-if="showConfigModal"
      @mousedown="handleModalMouseDown"
      @mouseup="handleModalMouseUp"
      class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center"
      style="z-index: 9999"
    >
      <div
        class="bg-white dark:bg-gray-900 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto"
      >
        <div
          class="sticky top-0 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between"
        >
          <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Chart Configuration
          </h3>
          <button
            @click="showConfigModal = false"
            class="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <svg
              class="w-5 h-5 text-gray-500 dark:text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <div class="p-6 space-y-6">
          <div>
            <h4
              class="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3"
            >
              X-Axis Range
            </h4>
            <div class="space-y-3">
              <label class="flex items-center gap-2">
                <input
                  v-model="config.xRange.auto"
                  type="checkbox"
                  class="rounded border-gray-300 dark:border-gray-600 text-primary focus:ring-primary"
                />
                <span class="text-sm text-gray-700 dark:text-gray-300"
                  >Auto Range</span
                >
              </label>
              <div v-if="!config.xRange.auto" class="grid grid-cols-2 gap-3">
                <div>
                  <label
                    class="block text-xs text-gray-600 dark:text-gray-400 mb-1"
                    >Min</label
                  >
                  <input
                    v-model.number="config.xRange.min"
                    type="number"
                    class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label
                    class="block text-xs text-gray-600 dark:text-gray-400 mb-1"
                    >Max</label
                  >
                  <input
                    v-model.number="config.xRange.max"
                    type="number"
                    class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  />
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4
              class="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3"
            >
              Y-Axis Range
            </h4>
            <div class="space-y-3">
              <label class="flex items-center gap-2">
                <input
                  v-model="config.yRange.auto"
                  type="checkbox"
                  class="rounded border-gray-300 dark:border-gray-600 text-primary focus:ring-primary"
                />
                <span class="text-sm text-gray-700 dark:text-gray-300"
                  >Auto Range</span
                >
              </label>
              <div v-if="!config.yRange.auto" class="grid grid-cols-2 gap-3">
                <div>
                  <label
                    class="block text-xs text-gray-600 dark:text-gray-400 mb-1"
                    >Min</label
                  >
                  <input
                    v-model.number="config.yRange.min"
                    type="number"
                    step="0.1"
                    class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label
                    class="block text-xs text-gray-600 dark:text-gray-400 mb-1"
                    >Max</label
                  >
                  <input
                    v-model.number="config.yRange.max"
                    type="number"
                    step="0.1"
                    class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  />
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4
              class="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3"
            >
              Smoothing
            </h4>
            <div class="space-y-3">
              <div>
                <label
                  class="block text-xs text-gray-600 dark:text-gray-400 mb-1"
                  >Mode</label
                >
                <select
                  v-model="config.smoothingMode"
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                >
                  <option value="disabled">Disabled</option>
                  <option value="ema">EMA (Exponential Moving Average)</option>
                  <option value="ma">MA (Moving Average)</option>
                  <option value="gaussian">Gaussian</option>
                </select>
              </div>
              <div v-if="config.smoothingMode !== 'disabled'">
                <label
                  class="block text-xs text-gray-600 dark:text-gray-400 mb-1"
                >
                  <span v-if="config.smoothingMode === 'ema'">Decay (0-1)</span>
                  <span v-else-if="config.smoothingMode === 'ma'"
                    >Window Size (steps)</span
                  >
                  <span v-else-if="config.smoothingMode === 'gaussian'"
                    >Kernel Size</span
                  >
                </label>
                <input
                  v-model.number="config.smoothingValue"
                  type="number"
                  :step="config.smoothingMode === 'ema' ? 0.01 : 1"
                  :min="config.smoothingMode === 'ema' ? 0 : 1"
                  :max="config.smoothingMode === 'ema' ? 1 : 1000"
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                />
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span v-if="config.smoothingMode === 'ema'"
                    >Higher = smoother (0.9 recommended)</span
                  >
                  <span v-else-if="config.smoothingMode === 'ma'"
                    >Number of steps to average</span
                  >
                  <span v-else-if="config.smoothingMode === 'gaussian'"
                    >Larger = smoother (odd numbers work best)</span
                  >
                </p>
              </div>
              <div v-if="config.smoothingMode !== 'disabled'">
                <label class="flex items-center gap-2">
                  <input
                    v-model="config.showOriginal"
                    type="checkbox"
                    class="rounded border-gray-300 dark:border-gray-600 text-primary focus:ring-primary"
                  />
                  <span class="text-sm text-gray-700 dark:text-gray-300"
                    >Show Original Data (Dimmed)</span
                  >
                </label>
              </div>
            </div>
          </div>

          <div>
            <label
              class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-2"
            >
              Downsample Rate
            </label>
            <input
              v-model.number="config.downsampleRate"
              type="number"
              min="1"
              max="100"
              step="1"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            />
            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Skip every N steps (1 = no downsampling, 8-16 recommended for
              large datasets)
            </p>
          </div>

          <div>
            <label
              class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-2"
            >
              Line Width
            </label>
            <input
              v-model.number="config.lineWidth"
              type="range"
              min="0.5"
              max="5"
              step="0.5"
              class="w-full"
            />
            <div
              class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1"
            >
              <span>Thin</span>
              <span>Thick</span>
            </div>
          </div>

          <div>
            <label class="flex items-center gap-2">
              <input
                v-model="config.showMarkers"
                type="checkbox"
                class="rounded border-gray-300 dark:border-gray-600 text-primary focus:ring-primary"
              />
              <span class="text-sm text-gray-700 dark:text-gray-300"
                >Show Markers</span
              >
            </label>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Target only the spike lines, not the color indicators in tooltip */
.line-plot-container :deep(.hoverlayer > .spikeline),
.line-plot-container :deep(.hoverlayer > path[class*="spikeline"]),
.line-plot-container :deep(.hoverlayer > line[class*="spikeline"]) {
  stroke: #888888 !important;
  stroke-width: 1.5 !important;
  stroke-opacity: 0.6 !important;
}

/* Ensure tooltip color indicators keep their original colors */
.line-plot-container :deep(.hoverlayer .hovertext path) {
  stroke: unset !important;
  stroke-width: unset !important;
  stroke-opacity: unset !important;
}

/* Make tooltip follow cursor more closely */
.line-plot-container :deep(.hoverlayer .hovertext) {
  pointer-events: none;
}
</style>
