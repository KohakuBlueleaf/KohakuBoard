<script setup>
import { ref, onMounted, watch, onUnmounted } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps({
  data: {
    type: Object,
    required: true,
  },
  title: {
    type: String,
    default: "",
  },
  xaxis: {
    type: String,
    default: "Value",
  },
  yaxis: {
    type: String,
    default: "Frequency",
  },
  height: {
    type: Number,
    default: 400,
  },
});

const plotDiv = ref(null);
let resizeObserver = null;
const showConfigModal = ref(false);
const config = ref({
  xRange: { auto: true, min: null, max: null },
  yRange: { auto: true, min: null, max: null },
  bins: 30,
  barGap: 0.05,
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
});

watch(
  () => props.data,
  () => {
    createPlot();
  },
  { deep: true },
);

watch(
  config,
  () => {
    createPlot();
  },
  { deep: true },
);

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

  const trace = {
    type: "histogram",
    x: props.data.values,
    name: props.data.name || "Distribution",
    nbinsx: config.value.bins,
    marker: {
      color: isDarkMode()
        ? "rgba(96, 165, 250, 0.7)"
        : "rgba(59, 130, 246, 0.7)",
      line: {
        color: isDarkMode() ? "rgba(96, 165, 250, 1)" : "rgba(59, 130, 246, 1)",
        width: 1,
      },
    },
  };

  const xAxisConfig = config.value.xRange.auto
    ? { autorange: true }
    : { range: [config.value.xRange.min, config.value.xRange.max] };

  const yAxisConfig = config.value.yRange.auto
    ? { autorange: true }
    : { range: [config.value.yRange.min, config.value.yRange.max] };

  const layout = {
    xaxis: {
      title: { text: props.xaxis, font: { color: colors.text } },
      gridcolor: colors.grid,
      zerolinecolor: colors.zeroline,
      color: colors.text,
      ...xAxisConfig,
    },
    yaxis: {
      title: { text: props.yaxis, font: { color: colors.text } },
      gridcolor: colors.grid,
      zerolinecolor: colors.zeroline,
      color: colors.text,
      ...yAxisConfig,
    },
    height: props.height,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: colors.text },
    margin: { t: 10, r: 50, b: 50, l: 60 },
    bargap: config.value.barGap,
    hoverlabel: {
      bgcolor: isDarkMode() ? "#1f2937" : "#ffffff",
      bordercolor: colors.grid,
      font: { color: colors.text },
    },
  };

  const plotConfig = {
    responsive: true,
    displayModeBar: false,
    displaylogo: false,
  };

  Plotly.react(plotDiv.value, [trace], layout, plotConfig);
}

function resetView() {
  config.value.xRange.auto = true;
  config.value.yRange.auto = true;
}

function exportPNG() {
  Plotly.downloadImage(plotDiv.value, {
    format: "png",
    filename: props.title || "histogram",
    height: props.height,
    width: 1200,
    scale: 2,
  });
}

function openConfig() {
  showConfigModal.value = true;
}

function setupResizeObserver() {
  if (!plotDiv.value) return;

  resizeObserver = new ResizeObserver(() => {
    if (plotDiv.value) {
      Plotly.Plots.resize(plotDiv.value);
    }
  });

  resizeObserver.observe(plotDiv.value);
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
</script>

<template>
  <div class="relative">
    <div
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
      @click.self="showConfigModal = false"
      class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
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
                    v-model.number="config.xRange.max"
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
                    step="1"
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
                    step="1"
                    class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  />
                </div>
              </div>
            </div>
          </div>

          <div>
            <label
              class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-2"
            >
              Number of Bins
            </label>
            <input
              v-model.number="config.bins"
              type="range"
              min="10"
              max="100"
              step="5"
              class="w-full"
            />
            <div
              class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1"
            >
              <span>Fewer bins</span>
              <span class="font-medium">{{ config.bins }}</span>
              <span>More bins</span>
            </div>
          </div>

          <div>
            <label
              class="block text-sm font-medium text-gray-900 dark:text-gray-100 mb-2"
            >
              Bar Gap ({{ (config.barGap * 100).toFixed(0) }}%)
            </label>
            <input
              v-model.number="config.barGap"
              type="range"
              min="0"
              max="0.3"
              step="0.01"
              class="w-full"
            />
            <div
              class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1"
            >
              <span>No gap</span>
              <span>Wide gap</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
