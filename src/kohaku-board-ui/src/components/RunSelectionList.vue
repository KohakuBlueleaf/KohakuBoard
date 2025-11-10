<script setup>
const props = defineProps({
  runs: Array,
  selectedRunIds: Set,
  displayedRunIds: Array,
  runColors: Object,
  project: String,
});

const emit = defineEmits(["toggle", "update-color"]);

function toggleVisibility(runId, event) {
  event.preventDefault();
  event.stopPropagation();
  emit("toggle", runId);
}

function isDisplayed(runId) {
  return props.displayedRunIds.includes(runId);
}

function handleColorChange(runId, event) {
  event.stopPropagation();
  const color = event.target.value;
  emit("update-color", runId, color);
}

const formatAnnotation = (run) => {
  const label =
    run.annotation && run.annotation.trim().length > 0
      ? run.annotation
      : "No annotation";
  return `${label}(${run.run_id})`;
};
</script>

<template>
  <div class="run-selection-list">
    <router-link
      v-for="run in runs"
      :key="run.run_id"
      :to="`/projects/${project}/${run.run_id}`"
      class="run-item block px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-800 border-b border-gray-200 dark:border-gray-700 no-underline"
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
          <div class="flex flex-col min-w-0">
            <span
              class="text-sm font-medium truncate text-gray-900 dark:text-gray-100"
            >
              {{ run.name || run.run_id }}
            </span>
            <span class="text-xs truncate text-gray-500 dark:text-gray-400">
              {{ formatAnnotation(run) }}
            </span>
          </div>
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
    </router-link>
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
