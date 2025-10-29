<script setup>
const props = defineProps({
  runs: Array,
  selectedRunIds: Set,
  displayedRunIds: Array,
  runColors: Object,
  project: String,
});

const emit = defineEmits(["toggle", "update-color"]);
const router = useRouter();

function toggleVisibility(runId, event) {
  event.stopPropagation();
  emit("toggle", runId);
}

function isDisplayed(runId) {
  return props.displayedRunIds.includes(runId);
}

function goToRun(run) {
  router.push(`/projects/${props.project}/${run.run_id}`);
}

function handleColorChange(runId, event) {
  event.stopPropagation();
  const color = event.target.value;
  emit("update-color", runId, color);
}
</script>

<template>
  <div class="run-selection-list">
    <div
      v-for="run in runs"
      :key="run.run_id"
      class="run-item px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-800 border-b border-gray-200 dark:border-gray-700 cursor-pointer"
      @click="goToRun(run)"
    >
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-2 flex-1 min-w-0">
          <!-- Color picker (hidden input triggered by clicking the dot) -->
          <div class="relative flex-shrink-0">
            <input
              type="color"
              :value="runColors[run.run_id] || '#ccc'"
              @input="handleColorChange(run.run_id, $event)"
              class="absolute inset-0 w-3 h-3 opacity-0 cursor-pointer"
              @click.stop
            />
            <span
              class="w-3 h-3 rounded-full block cursor-pointer border border-gray-300 dark:border-gray-600"
              :style="{ backgroundColor: runColors[run.run_id] || '#ccc' }"
              :title="'Click to change color'"
            ></span>
          </div>
          <span class="text-sm truncate">{{ run.name || run.run_id }}</span>
        </div>

        <el-button
          :type="selectedRunIds.has(run.run_id) ? 'primary' : 'default'"
          size="small"
          circle
          @click="toggleVisibility(run.run_id, $event)"
          :title="
            selectedRunIds.has(run.run_id)
              ? 'Visible in charts'
              : 'Hidden from charts'
          "
        >
          <i
            :class="selectedRunIds.has(run.run_id) ? 'i-ep-view' : 'i-ep-hide'"
          ></i>
        </el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.run-selection-list {
  overflow-y: auto;
}

.run-item {
  cursor: pointer;
}
</style>
