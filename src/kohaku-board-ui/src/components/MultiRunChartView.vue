<script setup>
import { ElMessage } from "element-plus";
import { Loading } from "@element-plus/icons-vue";
import MultiRunLineChart from "./MultiRunLineChart.vue";

const props = defineProps({
  project: String,
  displayedRuns: Array, // Max 10 runs to display
  chartsPerPage: Number, // 6, 8, or 12
  runColors: Object,
});

const runSummaries = ref({});
const scalarData = ref({});
const aggregatedMetrics = ref({
  scalars: [],
  media: new Map(),
  tables: new Map(),
  histograms: new Map(),
});
const tabs = ref([{ name: "All Metrics", metrics: [] }]);
const activeTab = ref("All Metrics");
const currentChartPage = ref(0);
const loading = ref(false);

// Fetch summaries for displayed runs
async function fetchSummaries() {
  if (!props.displayedRuns || props.displayedRuns.length === 0) {
    runSummaries.value = {};
    return;
  }

  loading.value = true;
  try {
    const runIds = props.displayedRuns.map((r) => r.run_id);
    const response = await fetch(
      `/api/projects/${props.project}/runs/batch/summary`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: runIds }),
      },
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch summaries: ${response.status}`);
    }

    runSummaries.value = await response.json();
    aggregateMetrics();
  } catch (error) {
    console.error("Failed to fetch summaries:", error);
    ElMessage.error("Failed to fetch run summaries");
  } finally {
    loading.value = false;
  }
}

// Aggregate metrics across all displayed runs and build tabs
function aggregateMetrics() {
  const scalars = new Set();
  const media = new Map();
  const tables = new Map();
  const histograms = new Map();

  for (const run of props.displayedRuns) {
    const summary = runSummaries.value[run.run_id];
    if (!summary) continue;

    const availableData = summary.available_data;

    // Scalars
    for (const metric of availableData.scalars || []) {
      scalars.add(metric);
    }

    // Media
    for (const metric of availableData.media || []) {
      if (!media.has(metric)) {
        media.set(metric, []);
      }
      media.get(metric).push(run);
    }

    // Tables
    for (const metric of availableData.tables || []) {
      if (!tables.has(metric)) {
        tables.set(metric, []);
      }
      tables.get(metric).push(run);
    }

    // Histograms
    for (const metric of availableData.histograms || []) {
      if (!histograms.has(metric)) {
        histograms.set(metric, []);
      }
      histograms.get(metric).push(run);
    }
  }

  aggregatedMetrics.value = {
    scalars: Array.from(scalars).sort(),
    media,
    tables,
    histograms,
  };

  // Build tabs based on metric namespaces
  buildTabs();

  // Fetch scalar data for all aggregated metrics
  fetchScalarData();
}

// Build tabs from metrics (group by namespace like run page)
function buildTabs() {
  const tabMap = new Map();
  tabMap.set("", []); // Main tab

  // Group scalars by namespace
  for (const metric of aggregatedMetrics.value.scalars) {
    const slashIdx = metric.indexOf("/");
    if (slashIdx > 0) {
      const namespace = metric.substring(0, slashIdx);
      if (!tabMap.has(namespace)) {
        tabMap.set(namespace, []);
      }
      tabMap.get(namespace).push(metric);
    } else {
      tabMap.get("").push(metric);
    }
  }

  // Build tabs array
  const newTabs = [];

  // Main tab
  if (tabMap.get("").length > 0) {
    newTabs.push({
      name: "Metrics",
      metrics: tabMap.get(""),
    });
  }

  // Namespace tabs
  for (const [namespace, metrics] of tabMap.entries()) {
    if (namespace !== "" && metrics.length > 0) {
      newTabs.push({
        name: namespace,
        metrics,
      });
    }
  }

  tabs.value = newTabs;
  if (newTabs.length > 0) {
    activeTab.value = newTabs[0].name;
  }
}

// Fetch scalar data for all runs × metrics
async function fetchScalarData() {
  if (aggregatedMetrics.value.scalars.length === 0) {
    scalarData.value = {};
    return;
  }

  loading.value = true;
  try {
    const runIds = props.displayedRuns.map((r) => r.run_id);
    const response = await fetch(
      `/api/projects/${props.project}/runs/batch/scalars`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_ids: runIds,
          metrics: aggregatedMetrics.value.scalars,
        }),
      },
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch scalars: ${response.status}`);
    }

    scalarData.value = await response.json();
  } catch (error) {
    console.error("Failed to fetch scalar data:", error);
    ElMessage.error("Failed to fetch metric data");
  } finally {
    loading.value = false;
  }
}

// Get metrics for current tab
const currentTabMetrics = computed(() => {
  const tab = tabs.value.find((t) => t.name === activeTab.value);
  return tab ? tab.metrics : [];
});

// Build chart list for current tab
const scalarCharts = computed(() => {
  const charts = [];

  for (const metric of currentTabMetrics.value) {
    // Build runs array with data for this metric
    const runsWithData = [];

    for (const run of props.displayedRuns) {
      const runData = scalarData.value[run.run_id];
      if (runData && runData[metric]) {
        const displayName = run.name
          ? `${run.run_id} · ${run.name}`
          : run.run_id;

        runsWithData.push({
          run_id: run.run_id,
          name: displayName,
          color: props.runColors[run.run_id] || "#ccc",
          data: runData[metric],
        });
      }
    }

    if (runsWithData.length > 0) {
      charts.push({
        type: "line",
        metricName: metric,
        runs: runsWithData,
      });
    }
  }

  return charts;
});

// Paginated charts
const visibleCharts = computed(() => {
  const start = currentChartPage.value * props.chartsPerPage;
  return scalarCharts.value.slice(start, start + props.chartsPerPage);
});

const totalPages = computed(() =>
  Math.ceil(scalarCharts.value.length / props.chartsPerPage),
);

// Watch for changes to displayed runs
watch(() => props.displayedRuns, fetchSummaries, {
  deep: true,
  immediate: true,
});

// Watch for tab changes - reset pagination
watch(activeTab, () => {
  currentChartPage.value = 0;
});

function nextPage() {
  if (currentChartPage.value < totalPages.value - 1) {
    currentChartPage.value++;
  }
}

function prevPage() {
  if (currentChartPage.value > 0) {
    currentChartPage.value--;
  }
}
</script>

<template>
  <div class="multi-run-chart-view p-4">
    <div v-if="loading" class="loading text-center py-10">
      <el-icon class="is-loading"><Loading /></el-icon>
      <p class="mt-2">Loading data...</p>
    </div>

    <div
      v-else-if="scalarCharts.length === 0"
      class="no-data text-center py-10"
    >
      <p class="text-gray-500 dark:text-gray-400">
        {{
          displayedRuns.length === 0
            ? "Select runs to compare"
            : "No metrics available"
        }}
      </p>
    </div>

    <div v-else>
      <!-- Tabs for metric namespaces -->
      <el-tabs v-model="activeTab" type="card" class="mb-4">
        <el-tab-pane
          v-for="tab in tabs"
          :key="tab.name"
          :label="tab.name"
          :name="tab.name"
        >
          <template #label>
            <span>{{ tab.name }} ({{ tab.metrics.length }})</span>
          </template>
        </el-tab-pane>
      </el-tabs>

      <!-- Pagination controls -->
      <div class="pagination-controls flex items-center justify-between mb-4">
        <div class="text-sm text-gray-600 dark:text-gray-400">
          Showing {{ visibleCharts.length }} of {{ scalarCharts.length }} charts
          (Page {{ currentChartPage + 1 }} / {{ totalPages }})
        </div>

        <div class="flex gap-2">
          <el-button
            size="small"
            :disabled="currentChartPage === 0"
            @click="prevPage"
            icon="ArrowLeft"
          >
            Previous
          </el-button>
          <el-button
            size="small"
            :disabled="currentChartPage >= totalPages - 1"
            @click="nextPage"
            icon="ArrowRight"
          >
            Next
          </el-button>
        </div>
      </div>

      <!-- Charts grid (only line charts) -->
      <div class="charts-grid grid gap-4">
        <div
          v-for="(chart, index) in visibleCharts"
          :key="`${chart.metricName}-${index}`"
          class="chart-container"
        >
          <MultiRunLineChart
            :metric-name="chart.metricName"
            :runs="chart.runs"
            x-metric="step"
            :height="400"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.charts-grid {
  grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
}

.chart-container {
  min-height: 400px;
}

@media (max-width: 768px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
}
</style>
