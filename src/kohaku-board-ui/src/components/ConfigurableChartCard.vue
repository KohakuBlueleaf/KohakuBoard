<script setup>
import LinePlot from "./LinePlot.vue";
import MediaViewer from "./MediaViewer.vue";
import HistogramViewer from "./HistogramViewer.vue";
import TableViewer from "./TableViewer.vue";

const props = defineProps({
  cardId: String,
  sparseData: Object,
  availableMetrics: Array,
  initialConfig: Object,
  experimentId: String,
  project: String, // Optional: if provided, use runs API instead of experiments API
  tabName: String,
  hoverSyncEnabled: {
    type: Boolean,
    default: true,
  },
  multiRunMode: {
    type: Boolean,
    default: false,
  },
  runColors: Object, // Map of run_id -> color (for multi-run mode)
  runNames: Object, // Map of run_id -> name (for multi-run mode)
});

const emit = defineEmits(["update:config", "remove"]);

console.log(`[${props.cardId}] Component created`);
console.log(`[${props.cardId}] Hover sync props:`, {
  tabName: props.tabName,
  hoverSyncEnabled: props.hoverSyncEnabled,
});

// Direct reference to props
const cfg = props.initialConfig;
const localHeight = ref(cfg.height);
const localWidth = ref(cfg.widthPercent);
const isEditingTitle = ref(false);
const editedTitle = ref(cfg.title);
const isResizingWidth = ref(false);
const previewWidth = ref(null);
const showSettings = ref(false);
const plotRef = ref(null);

// Sync when parent updates (only if actually different)
watch(
  () => props.initialConfig.height,
  (h, oldH) => {
    if (h !== undefined && h !== localHeight.value) {
      console.log(
        `[${props.cardId}] Height prop changed: ${localHeight.value} → ${h}`,
      );
      localHeight.value = h;
    }
  },
);
watch(
  () => props.initialConfig.widthPercent,
  (w, oldW) => {
    if (w !== undefined && w !== localWidth.value) {
      console.log(
        `[${props.cardId}] Width prop changed: ${localWidth.value} → ${w}`,
      );
      localWidth.value = w;
    }
  },
);

onMounted(() => {
  console.log(`[${props.cardId}] Component mounted`);
});

onUnmounted(() => {
  console.log(`[${props.cardId}] Component unmounted`);
});

onUpdated(() => {
  console.log(`[${props.cardId}] Component updated/re-rendered`);
});

const cardType = computed(() => props.initialConfig.type || "line");
const mediaData = ref(null);
const histogramData = ref(null);
const tableData = ref(null);
const currentStepIndex = ref(0);
const isCardLoading = ref(false);

// Fetch non-scalar data based on type
watch(
  cardType,
  async (type) => {
    isCardLoading.value = true;
    try {
      // Use runs API if project is provided, otherwise use experiments API
      const baseUrl = props.project
        ? `/api/projects/${props.project}/runs/${props.experimentId}`
        : `/api/experiments/${props.experimentId}`;

      if (type === "media" && props.initialConfig.mediaName) {
        const res = await fetch(
          `${baseUrl}/media/${props.initialConfig.mediaName}`,
        );
        const data = await res.json();
        mediaData.value = data.data;
      } else if (type === "histogram" && props.initialConfig.histogramName) {
        const res = await fetch(
          `${baseUrl}/histograms/${props.initialConfig.histogramName}`,
        );
        const data = await res.json();
        histogramData.value = data.data;
      } else if (type === "table" && props.initialConfig.tableName) {
        const res = await fetch(
          `${baseUrl}/tables/${props.initialConfig.tableName}`,
        );
        const data = await res.json();
        tableData.value = data.data;
      }
    } finally {
      isCardLoading.value = false;
    }
  },
  { immediate: true },
);

const hasDataError = computed(() => {
  const config = props.initialConfig;
  console.log(
    `[${props.cardId}] hasDataError computed, cardType: ${cardType.value}`,
  );

  if (cardType.value !== "line") return false;
  if (!config.xMetric || !config.yMetrics || config.yMetrics.length === 0)
    return false;

  // In multi-run mode, x-axis data is per-run (e.g., "global_step (run_id)")
  // Check if at least one run has x-axis data
  let hasAnyXData = false;
  if (props.multiRunMode) {
    for (const key of Object.keys(props.sparseData)) {
      if (key.startsWith(config.xMetric + " (")) {
        const xData = props.sparseData[key];
        if (xData && xData.length > 0 && xData.some((v) => v !== null)) {
          hasAnyXData = true;
          break;
        }
      }
    }
    console.log(
      `[${props.cardId}] hasDataError - multi-run xData check: ${hasAnyXData}`,
    );
  } else {
    const xData = props.sparseData[config.xMetric];
    hasAnyXData = xData && xData.length > 0;
    console.log(
      `[${props.cardId}] hasDataError - single-run xData:`,
      xData ? `${xData.length} elements` : "missing",
    );
  }

  if (!hasAnyXData) {
    console.log(`[${props.cardId}] hasDataError = true (no xData)`);
    return true;
  }

  // In multi-run mode, need to expand base metrics to find actual data keys
  let metricsToCheck = config.yMetrics;
  if (props.multiRunMode) {
    metricsToCheck = [];
    for (const baseMetric of config.yMetrics) {
      for (const key of Object.keys(props.sparseData)) {
        if (key.startsWith(baseMetric + " (") || key === baseMetric) {
          metricsToCheck.push(key);
        }
      }
    }
    console.log(
      `[${props.cardId}] hasDataError - expanded metrics:`,
      metricsToCheck,
    );
  }

  // Check if all y metrics have no valid data (including NaN/inf as valid!)
  const allEmpty = metricsToCheck.every((yMetric) => {
    const yData = props.sparseData[yMetric];
    if (!yData || yData.length === 0) return true;

    // Check if there's at least ONE non-null value (including NaN/inf)
    const hasAnyData = yData.some((val) => val !== null);
    return !hasAnyData;
  });

  console.log(`[${props.cardId}] hasDataError = ${allEmpty}`);
  return allEmpty;
});

const processedChartData = computed(() => {
  const config = props.initialConfig;
  console.log(
    `[${props.cardId}] processedChartData computed START`,
    `type: ${cardType.value}`,
    `multiRunMode: ${props.multiRunMode}`,
    `xMetric: ${config.xMetric}`,
    `yMetrics:`,
    config.yMetrics,
    `sparseData keys:`,
    Object.keys(props.sparseData),
  );

  if (cardType.value !== "line") {
    console.log(`[${props.cardId}] Not line type, returning empty`);
    return [];
  }
  if (!config.xMetric || !config.yMetrics || config.yMetrics.length === 0) {
    console.log(
      `[${props.cardId}] Missing xMetric or yMetrics, returning empty`,
    );
    return [];
  }

  // In single-run mode, x-data is shared
  // In multi-run mode, x-data is per-run (checked later)
  const xData = props.sparseData[config.xMetric];

  if (!props.multiRunMode) {
    console.log(
      `[${props.cardId}] Single-run mode - xData for ${config.xMetric}:`,
      xData ? `${xData.length} elements` : "null/undefined",
    );
    if (!xData) {
      console.log(`[${props.cardId}] No xData, returning empty`);
      return [];
    }
  } else {
    console.log(`[${props.cardId}] Multi-run mode - will use per-run xData`);
  }

  // In multi-run mode, expand each base metric to all runs
  let expandedYMetrics = config.yMetrics;
  if (props.multiRunMode) {
    expandedYMetrics = [];
    for (const baseMetric of config.yMetrics) {
      // Find all keys in sparseData that match this metric (with any run suffix)
      for (const key of Object.keys(props.sparseData)) {
        if (key.startsWith(baseMetric + " (") || key === baseMetric) {
          expandedYMetrics.push(key);
        }
      }
    }
    console.log(
      `[${props.cardId}] Multi-run mode: base metrics:`,
      config.yMetrics,
    );
    console.log(
      `[${props.cardId}] Expanded to ${expandedYMetrics.length} series:`,
      expandedYMetrics,
    );
    console.log(
      `[${props.cardId}] Available in sparseData:`,
      Object.keys(props.sparseData),
    );
  }

  return expandedYMetrics
    .map((yMetric) => {
      const yData = props.sparseData[yMetric];
      if (!yData || yData.length === 0) return null; // Handle empty data gracefully

      // In multi-run mode, each y-metric has a corresponding x-metric with same run suffix
      // E.g., "train/loss (run1)" uses "global_step (run1)"
      let xDataToUse = xData;
      if (props.multiRunMode && yMetric.includes(" (")) {
        // Extract run_id from y-metric: "train/loss (run_id)" -> "run_id"
        const runIdMatch = yMetric.match(/\(([^)]+)\)$/);
        if (runIdMatch) {
          const runId = runIdMatch[1];
          const xMetricWithRun = `${config.xMetric} (${runId})`;
          xDataToUse = props.sparseData[xMetricWithRun] || xData;
          console.log(
            `[${props.cardId}] Using x-data ${xMetricWithRun} for ${yMetric}`,
          );
        }
      }

      const x = [];
      const y = [];
      let lastXValue = null;
      let nanCount = 0;
      let infCount = 0;
      let negInfCount = 0;

      for (let i = 0; i < xDataToUse.length; i++) {
        let xVal = xDataToUse[i];
        const yVal = yData[i];

        // Convert timestamp strings to Date objects for Plotly
        if (config.xMetric === "timestamp" && typeof xVal === "string") {
          xVal = new Date(xVal);
        }
        // Convert walltime (unix seconds) to Date objects for Plotly
        else if (config.xMetric === "walltime" && typeof xVal === "number") {
          xVal = new Date(xVal * 1000); // Convert seconds to milliseconds
        }

        if (xVal !== null) lastXValue = xVal;

        // Include NaN/inf values (they are not null!)
        // Only skip if yVal is actually null (missing data)
        if (yVal !== null && lastXValue !== null) {
          x.push(lastXValue);
          y.push(yVal);

          // Count special values for logging
          if (isNaN(yVal)) {
            nanCount++;
          } else if (yVal === Infinity) {
            infCount++;
          } else if (yVal === -Infinity) {
            negInfCount++;
          }
        }
      }

      const specialValuesLog = [];
      if (nanCount > 0) specialValuesLog.push(`NaN=${nanCount}`);
      if (infCount > 0) specialValuesLog.push(`+inf=${infCount}`);
      if (negInfCount > 0) specialValuesLog.push(`-inf=${negInfCount}`);

      console.log(
        `[${props.cardId}] ${yMetric}: collected ${y.length} points` +
          (specialValuesLog.length > 0
            ? ` (${specialValuesLog.join(", ")})`
            : ""),
      );

      // Return series even if all values are NaN/inf (x/y might be empty but we still need the series)
      // The LinePlot component will handle special values
      return { name: yMetric, x, y };
    })
    .filter((d) => d !== null);
});

function saveTitle() {
  const newConfig = { ...props.initialConfig, title: editedTitle.value };
  isEditingTitle.value = false;
  emit("update:config", { id: props.cardId, config: newConfig });
}

function emitConfig(updates = {}) {
  const newConfig = { ...props.initialConfig, ...updates };
  console.log(`[${props.cardId}] emitConfig:`, updates, "→", newConfig);
  console.log(`[${props.cardId}] Emitting update:config to parent`);
  emit("update:config", { id: props.cardId, config: newConfig });
}

function resetView() {
  console.log(
    `[${props.cardId}] resetView called, plotRef:`,
    !!plotRef.value,
    "has resetView:",
    !!plotRef.value?.resetView,
  );
  if (plotRef.value?.resetView) {
    console.log(`[${props.cardId}] Calling plotRef.resetView()`);
    plotRef.value.resetView();
  } else {
    console.warn(`[${props.cardId}] plotRef.resetView not available`);
  }
}

function exportPNG() {
  if (plotRef.value?.exportPNG) {
    plotRef.value.exportPNG();
  }
}

const plotConfig = computed(() => plotRef.value?.plotConfig || null);

// Watch plotConfig and save entire config immediately when ANY setting changes
watch(
  plotConfig,
  (newConfig) => {
    if (!newConfig) return;

    // Save entire config whenever anything changes
    emitConfig({
      smoothingMode: newConfig.smoothingMode,
      smoothingValue: newConfig.smoothingValue,
      downsampleRate: newConfig.downsampleRate,
      showOriginal: newConfig.showOriginal,
      lineWidth: newConfig.lineWidth,
      showMarkers: newConfig.showMarkers,
      xRange: newConfig.xRange,
      yRange: newConfig.yRange,
    });
  },
  { deep: true },
);

function startResizeBottom(e) {
  e.preventDefault();
  e.stopPropagation();
  const startY = e.clientY;
  const startHeight = props.initialConfig.height;
  let tempHeight = startHeight;
  const shiftWasPressed = e.shiftKey;

  const onMove = (e) => {
    tempHeight = Math.max(
      200,
      Math.min(1000, startHeight + (e.clientY - startY)),
    );
    localHeight.value = tempHeight;

    // If shift pressed, emit realtime sync
    if (shiftWasPressed) {
      emit("update:config", {
        id: props.cardId,
        config: { ...props.initialConfig, height: tempHeight },
        syncAll: true,
        realtime: true, // Flag for realtime updates during drag
      });
    }
  };

  const onUp = () => {
    // Final update
    if (shiftWasPressed) {
      emit("update:config", {
        id: props.cardId,
        config: { ...props.initialConfig, height: tempHeight },
        syncAll: true,
        realtime: false,
      });
    } else {
      emitConfig({ height: tempHeight });
    }
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  };

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

function startResizeRight(e) {
  e.preventDefault();
  e.stopPropagation();

  isResizingWidth.value = true;
  const shiftWasPressed = e.shiftKey;
  const startX = e.clientX;
  const cardEl = e.target.closest(".chart-card-wrapper");
  if (!cardEl) return;

  const parentWidth =
    cardEl.parentElement?.getBoundingClientRect().width || 1000;
  const startWidth = cardEl.getBoundingClientRect().width;

  // Initialize preview with current width
  previewWidth.value = props.initialConfig.widthPercent;

  const onMove = (e) => {
    const deltaX = e.clientX - startX;
    const newWidth = startWidth + deltaX;
    const ratioPercent = (newWidth / parentWidth) * 100;

    const snaps = [
      { p: 12.5, l: "1/8" },
      { p: 16.666666666666666666666666666, l: "1/6" },
      { p: 20.0, l: "1/5" },
      { p: 25.0, l: "1/4" },
      { p: 33.333333333333333333333333333, l: "1/3" },
      { p: 50.0, l: "1/2" },
      { p: 66.666666666666666666666666666, l: "2/3" },
      { p: 100.0, l: "Full" },
    ];

    let closest = snaps[0];
    let minDiff = Math.abs(ratioPercent - snaps[0].p);

    for (const snap of snaps) {
      const diff = Math.abs(ratioPercent - snap.p);
      if (diff < minDiff) {
        minDiff = diff;
        closest = snap;
      }
    }

    previewWidth.value = closest.p;
    localWidth.value = closest.p;

    // If shift pressed, emit realtime sync
    if (shiftWasPressed) {
      emit("update:config", {
        id: props.cardId,
        config: { ...props.initialConfig, widthPercent: closest.p },
        syncAll: true,
        realtime: true,
      });
    }
  };

  const onUp = () => {
    if (
      previewWidth.value !== null &&
      previewWidth.value !== props.initialConfig.widthPercent
    ) {
      console.log(`[${props.cardId}] Width changed, shift=${shiftWasPressed}`);
      // Final update
      if (shiftWasPressed) {
        emit("update:config", {
          id: props.cardId,
          config: { widthPercent: localWidth.value, height: localHeight.value },
          syncAll: true,
          realtime: false,
        });
      } else {
        emitConfig({
          widthPercent: localWidth.value,
          height: localHeight.value,
        });
      }
    } else {
      console.log(
        `[${props.cardId}] Width unchanged (${previewWidth.value}), not emitting`,
      );
    }
    isResizingWidth.value = false;
    previewWidth.value = null;
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  };

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}
</script>

<template>
  <div
    class="chart-card-wrapper"
    :class="{ 'resizing-width': isResizingWidth }"
    :style="{
      flex: `0 0 calc(${localWidth || 100}% - 16px)`,
      maxWidth: `calc(${localWidth || 100}% - 16px)`,
    }"
  >
    <div
      v-if="isResizingWidth"
      class="width-preview-overlay"
      :style="{
        width: `calc(${previewWidth}% - 16px)`,
      }"
    >
      <div class="width-preview-text">
        {{
          previewWidth === 100
            ? "Full"
            : previewWidth === 66.666666666666666666666666666
              ? "2/3"
              : previewWidth === 50
                ? "1/2"
                : previewWidth === 33.333333333333333333333333333
                  ? "1/3"
                  : previewWidth === 25
                    ? "1/4"
                    : previewWidth === 20
                      ? "1/5"
                      : previewWidth === 16.666666666666666666666666666
                        ? "1/6"
                        : previewWidth === 12.5
                          ? "1/8"
                          : previewWidth + "%"
        }}
      </div>
    </div>
    <el-card :body-style="{ padding: '12px' }">
      <template #header>
        <div class="space-y-2">
          <div class="flex items-center gap-2 card-drag-handle cursor-move">
            <i class="i-ep-rank text-gray-400"></i>
            <span
              v-if="!isEditingTitle"
              class="font-semibold truncate flex-1 cursor-pointer"
              :title="props.initialConfig.title"
              @dblclick="isEditingTitle = true"
            >
              {{ props.initialConfig.title }}
            </span>
            <el-input
              v-else
              v-model="editedTitle"
              size="small"
              class="flex-1"
              @keyup.enter="saveTitle"
              @keyup.esc="isEditingTitle = false"
              @click.stop
            />
            <el-button
              v-if="!isEditingTitle"
              icon="Edit"
              size="small"
              text
              @click.stop="isEditingTitle = true"
            />
            <el-button
              v-if="!isEditingTitle"
              size="small"
              text
              type="danger"
              @click.stop="$emit('remove')"
              title="Remove Card"
            >
              <i class="i-ep-delete"></i>
            </el-button>
            <el-button
              v-else
              size="small"
              type="primary"
              @click.stop="saveTitle"
              >Save</el-button
            >
            <el-button
              v-if="isEditingTitle"
              size="small"
              @click.stop="isEditingTitle = false"
              >Cancel</el-button
            >
          </div>
          <div
            class="flex gap-0.5 border-t border-gray-100 dark:border-gray-700 pt-2"
          >
            <el-button size="small" @click.stop="resetView" title="Reset View">
              <i class="i-ep-refresh-left"></i>
            </el-button>
            <el-button size="small" @click.stop="exportPNG" title="Export PNG">
              <i class="i-ep-download"></i>
            </el-button>
            <el-button
              size="small"
              @click.stop="showSettings = true"
              title="Settings"
            >
              <i class="i-ep-setting"></i>
            </el-button>
          </div>
        </div>
      </template>

      <div
        class="plot-container relative"
        :style="{ height: `${localHeight}px` }"
      >
        <div
          v-if="isCardLoading"
          class="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-10"
        >
          <div class="text-gray-500 dark:text-gray-400">Loading...</div>
        </div>
        <div
          v-else-if="hasDataError && cardType === 'line'"
          class="absolute inset-0 flex items-center justify-center bg-white dark:bg-gray-900"
        >
          <div class="text-center">
            <div class="text-gray-500 dark:text-gray-400 mb-2">
              No data available
            </div>
            <div class="text-xs text-gray-400 dark:text-gray-500">
              Check metric name or data source
            </div>
          </div>
        </div>
        <LinePlot
          v-else-if="cardType === 'line' && processedChartData.length > 0"
          ref="plotRef"
          :data="processedChartData"
          :xaxis="props.initialConfig.xMetric"
          yaxis="Value"
          :height="localHeight"
          :hide-toolbar="true"
          :smoothing-mode="props.initialConfig.smoothingMode"
          :smoothing-value="props.initialConfig.smoothingValue"
          :downsample-rate="props.initialConfig.downsampleRate"
          :show-original="props.initialConfig.showOriginal"
          :line-width="props.initialConfig.lineWidth"
          :show-markers="props.initialConfig.showMarkers"
          :x-range="props.initialConfig.xRange"
          :y-range="props.initialConfig.yRange"
          :tab-name="props.tabName"
          :chart-id="props.cardId"
          :hover-sync-enabled="props.hoverSyncEnabled"
          :multi-run-mode="props.multiRunMode"
          :run-colors="props.runColors"
          :run-names="props.runNames"
        />
        <MediaViewer
          v-else-if="cardType === 'media' && mediaData"
          :media-data="mediaData"
          :height="localHeight"
          :current-step="
            props.initialConfig.currentStep ??
            (mediaData.length > 0 ? mediaData[mediaData.length - 1].step : 0)
          "
          :card-id="props.cardId"
          :auto-advance-to-latest="
            props.initialConfig.autoAdvanceToLatest !== false
          "
          @update:current-step="(s) => emitConfig({ currentStep: s })"
          @update:auto-advance="(v) => emitConfig({ autoAdvanceToLatest: v })"
        />
        <HistogramViewer
          v-else-if="cardType === 'histogram' && histogramData"
          ref="plotRef"
          :histogram-data="histogramData"
          :height="localHeight"
          :current-step="props.initialConfig.currentStep || 0"
          :card-id="props.cardId"
          :initial-mode="props.initialConfig.histogramMode || 'single'"
          :downsample-rate="props.initialConfig.downsampleRate || -1"
          :x-axis="props.initialConfig.histogramXAxis || 'global_step'"
          @update:current-step="(s) => emitConfig({ currentStep: s })"
          @update:mode="(m) => emitConfig({ histogramMode: m })"
          @update:x-axis="(x) => emitConfig({ histogramXAxis: x })"
        />
        <TableViewer
          v-else-if="cardType === 'table' && tableData"
          :table-data="tableData"
          :height="localHeight"
          :current-step="
            props.initialConfig.currentStep ??
            (tableData.length > 0 ? tableData[tableData.length - 1].step : 0)
          "
          :card-id="props.cardId"
          :auto-advance-to-latest="
            props.initialConfig.autoAdvanceToLatest !== false
          "
          @update:current-step="(s) => emitConfig({ currentStep: s })"
          @update:auto-advance="(v) => emitConfig({ autoAdvanceToLatest: v })"
        />
        <el-empty v-else description="Select metrics" />
      </div>
    </el-card>

    <div
      @mousedown.stop.prevent="startResizeBottom"
      class="resize-handle resize-handle-bottom"
      title="Drag to resize height"
    ></div>
    <div
      @mousedown.stop.prevent="startResizeRight"
      class="resize-handle resize-handle-right"
      title="Drag to resize width"
    ></div>
    <div
      @mousedown.stop.prevent="startResizeRight"
      class="resize-handle resize-handle-corner"
      title="Drag to resize width"
    ></div>

    <el-dialog
      v-model="showSettings"
      title="Chart Settings"
      width="700px"
      @click.stop
      append-to-body
    >
      <el-form label-width="150px">
        <el-divider content-position="left">Data Selection</el-divider>
        <el-form-item label="X-Axis">
          <el-select
            :model-value="props.initialConfig.xMetric"
            @change="(v) => emitConfig({ xMetric: v })"
            class="w-full"
          >
            <el-option
              v-for="m in availableMetrics"
              :key="m"
              :label="m"
              :value="m"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="Y-Axis">
          <el-select
            :model-value="props.initialConfig.yMetrics"
            @change="(v) => emitConfig({ yMetrics: v })"
            multiple
            collapse-tags
            collapse-tags-tooltip
            class="w-full"
            placeholder="Select metrics"
          >
            <el-option
              v-for="m in availableMetrics.filter(
                (x) => x !== props.initialConfig.xMetric,
              )"
              :key="m"
              :label="m"
              :value="m"
            />
          </el-select>
        </el-form-item>

        <template v-if="plotConfig">
          <el-divider content-position="left">Axis Range</el-divider>
          <el-row :gutter="12">
            <el-col :span="12">
              <el-form-item label="X-Axis Auto">
                <el-switch v-model="plotConfig.xRange.auto" />
              </el-form-item>
              <el-row v-if="!plotConfig.xRange.auto" :gutter="8">
                <el-col :span="12">
                  <el-input
                    v-model.number="plotConfig.xRange.min"
                    placeholder="Min"
                    size="small"
                    type="number"
                  />
                </el-col>
                <el-col :span="12">
                  <el-input
                    v-model.number="plotConfig.xRange.max"
                    placeholder="Max"
                    size="small"
                    type="number"
                  />
                </el-col>
              </el-row>
            </el-col>
            <el-col :span="12">
              <el-form-item label="Y-Axis Auto">
                <el-switch v-model="plotConfig.yRange.auto" />
              </el-form-item>
              <el-row v-if="!plotConfig.yRange.auto" :gutter="8">
                <el-col :span="12">
                  <el-input
                    v-model.number="plotConfig.yRange.min"
                    placeholder="Min"
                    size="small"
                    type="number"
                    step="0.1"
                  />
                </el-col>
                <el-col :span="12">
                  <el-input
                    v-model.number="plotConfig.yRange.max"
                    placeholder="Max"
                    size="small"
                    type="number"
                    step="0.1"
                  />
                </el-col>
              </el-row>
            </el-col>
          </el-row>

          <el-divider content-position="left">Smoothing</el-divider>
          <el-form-item label="Mode">
            <el-select v-model="plotConfig.smoothingMode" class="w-full">
              <el-option label="Disabled" value="disabled" />
              <el-option label="EMA (Exponential Moving Average)" value="ema" />
              <el-option label="MA (Moving Average)" value="ma" />
              <el-option label="Gaussian" value="gaussian" />
            </el-select>
          </el-form-item>
          <el-form-item
            v-if="plotConfig.smoothingMode !== 'disabled'"
            :label="
              plotConfig.smoothingMode === 'ema'
                ? 'Decay (0-1)'
                : plotConfig.smoothingMode === 'ma'
                  ? 'Window Size'
                  : 'Kernel Size'
            "
          >
            <el-input-number
              v-model="plotConfig.smoothingValue"
              :min="plotConfig.smoothingMode === 'ema' ? 0 : 1"
              :max="plotConfig.smoothingMode === 'ema' ? 1 : 1000"
              :step="plotConfig.smoothingMode === 'ema' ? 0.01 : 1"
              class="w-full"
            />
          </el-form-item>
          <el-form-item
            v-if="plotConfig.smoothingMode !== 'disabled'"
            label="Show Original"
          >
            <el-switch v-model="plotConfig.showOriginal" />
          </el-form-item>

          <el-divider content-position="left">Display Options</el-divider>
          <el-form-item label="Downsample Rate">
            <el-input-number
              v-model="plotConfig.downsampleRate"
              :min="1"
              :max="100"
              class="w-full"
            />
          </el-form-item>
          <el-form-item label="Line Width">
            <el-slider
              v-model="plotConfig.lineWidth"
              :min="0.5"
              :max="5"
              :step="0.5"
            />
          </el-form-item>
          <el-form-item label="Show Markers">
            <el-switch v-model="plotConfig.showMarkers" />
          </el-form-item>
        </template>
      </el-form>
    </el-dialog>
  </div>
</template>

<style scoped>
.chart-card-wrapper {
  position: relative;
  height: 100%;
}

@media (max-width: 900px) {
  .chart-card-wrapper {
    flex: 0 0 100% !important;
    max-width: 100% !important;
  }

  .chart-card-wrapper :deep(.el-card__body) {
    padding: 8px !important;
  }
}

.resize-handle {
  position: absolute;
  background: transparent;
  transition: background 0.2s;
}

.resize-handle:hover {
  background: rgba(64, 158, 255, 0.15);
}

.resize-handle-bottom {
  bottom: 0;
  left: 0;
  right: 0;
  height: 6px;
  cursor: ns-resize;
  z-index: 10;
}

.resize-handle-right {
  top: 0;
  right: 0;
  bottom: 0;
  width: 6px;
  cursor: ew-resize;
  z-index: 10;
}

.resize-handle-corner {
  bottom: 0;
  right: 0;
  width: 16px;
  height: 16px;
  cursor: nwse-resize;
  z-index: 10;
}

.resize-handle-corner::after {
  content: "";
  position: absolute;
  bottom: 1px;
  right: 1px;
  width: 0;
  height: 0;
  border-style: solid;
  border-width: 0 0 12px 12px;
  border-color: transparent transparent rgba(64, 158, 255, 0.4) transparent;
}

.resizing-width {
  opacity: 0.7;
}

.plot-container {
  position: relative;
  width: 100%;
}

.width-preview-overlay {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  background: rgba(64, 158, 255, 0.1);
  border: 3px dashed rgba(64, 158, 255, 0.5);
  z-index: 999;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  min-width: 100%;
}

.width-preview-text {
  background: rgba(64, 158, 255, 0.9);
  color: white;
  padding: 8px 16px;
  border-radius: 4px;
  font-weight: bold;
  font-size: 16px;
}
</style>
