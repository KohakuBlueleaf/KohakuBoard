<script setup>
import { computed, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";

const props = defineProps({
  metricName: String,
  runs: Array, // [{run_id, name, color, data: {steps, values}}]
  xMetric: { type: String, default: "step" },
  height: { type: Number, default: 400 },
});

const chartRef = ref(null);
const plotlyDiv = ref(null);

// Build Plotly traces
const traces = computed(() => {
  const result = [];

  for (const run of props.runs) {
    if (!run.data || !run.data.steps || !run.data.values) {
      continue;
    }

    const steps = run.data.steps;
    const values = run.data.values;

    // Filter out null values
    const xData = [];
    const yData = [];
    for (let i = 0; i < steps.length; i++) {
      if (values[i] !== null && values[i] !== undefined) {
        xData.push(steps[i]);
        yData.push(values[i]);
      }
    }

    if (xData.length === 0) continue;

    const runLabel = run.run_id;

    result.push({
      name: `${props.metricName} (${runLabel})`,
      x: xData,
      y: yData,
      type: "scatter",
      mode: "lines",
      line: { color: run.color, width: 2 },
      legendgroup: run.run_id,
    });
  }

  return result;
});

const layout = computed(() => ({
  title: props.metricName,
  xaxis: { title: props.xMetric },
  yaxis: { title: "Value" },
  height: props.height,
  margin: { l: 60, r: 40, t: 60, b: 60 },
  plot_bgcolor: "rgba(0,0,0,0)",
  paper_bgcolor: "rgba(0,0,0,0)",
  legend: {
    orientation: "v",
    x: 1.05,
    y: 1,
  },
}));

const config = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
};

watch(
  () => [traces.value, layout.value],
  () => {
    if (plotlyDiv.value && traces.value.length > 0) {
      Plotly.react(plotlyDiv.value, traces.value, layout.value, config);
    }
  },
  { deep: true, immediate: true },
);

onMounted(() => {
  if (plotlyDiv.value && traces.value.length > 0) {
    Plotly.newPlot(plotlyDiv.value, traces.value, layout.value, config);
  }
});

onUnmounted(() => {
  if (plotlyDiv.value) {
    Plotly.purge(plotlyDiv.value);
  }
});
</script>

<template>
  <div ref="chartRef" class="multi-run-line-chart">
    <div v-if="traces.length === 0" class="no-data">
      No data available for this metric
    </div>
    <div v-else ref="plotlyDiv"></div>
  </div>
</template>

<style scoped>
.multi-run-line-chart {
  width: 100%;
  background: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

:deep(.dark) .multi-run-line-chart {
  background: #1a1a1a;
}

.no-data {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 400px;
  color: #999;
}
</style>
