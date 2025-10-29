<script setup>
import { ref, onMounted } from "vue";
import { useRoute, useRouter } from "vue-router";
import { fetchProjectRuns } from "@/utils/api";

const route = useRoute();
const router = useRouter();
const project = ref(route.params.project);
const runs = ref([]);
const loading = ref(true);
const projectOwner = ref(null);

onMounted(async () => {
  try {
    const result = await fetchProjectRuns(project.value);
    runs.value = result.runs;
    projectOwner.value = result.owner;
  } catch (error) {
    console.error("Failed to fetch runs:", error);
  } finally {
    loading.value = false;
  }
});

function viewRun(runId) {
  // Route to the project-scoped experiment viewer
  router.push(`/projects/${project.value}/${runId}`);
}

function formatDate(timestamp) {
  if (!timestamp) return "N/A";
  return new Date(timestamp).toLocaleString();
}

function formatSize(bytes) {
  if (!bytes) return "N/A";
  const mb = bytes / 1024 / 1024;
  if (mb < 1) return `${(bytes / 1024).toFixed(1)} KB`;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-3xl font-bold text-gray-900 dark:text-gray-100">
          {{ project }}
        </h1>
        <p
          v-if="projectOwner"
          class="text-sm text-gray-500 dark:text-gray-400 mt-1"
        >
          Owner: {{ projectOwner }}
        </p>
      </div>
      <router-link
        to="/projects"
        class="px-4 py-2 bg-gray-600 hover:bg-gray-700 dark:bg-gray-500 dark:hover:bg-gray-600 text-white rounded-md font-medium transition-colors shadow-sm"
      >
        ← Back to Projects
      </router-link>
    </div>

    <div v-if="loading" class="text-center py-12">
      <div class="text-gray-500 dark:text-gray-400">Loading runs...</div>
    </div>

    <div
      v-else-if="runs.length === 0"
      class="bg-white dark:bg-gray-900 rounded-lg shadow-md p-8 text-center border border-gray-200 dark:border-gray-800"
    >
      <div class="text-gray-500 dark:text-gray-400 mb-4">No runs found</div>
      <p class="text-sm text-gray-400 dark:text-gray-500">
        This project doesn't have any training runs yet.
      </p>
    </div>

    <div v-else class="grid grid-cols-1 gap-4">
      <div
        v-for="run in runs"
        :key="run.run_id"
        @click="viewRun(run.run_id)"
        class="bg-white dark:bg-gray-900 rounded-lg shadow-md p-4 cursor-pointer hover:shadow-lg transition-all border border-gray-200 dark:border-gray-800 hover:border-blue-400 dark:hover:border-blue-600"
      >
        <div class="flex items-start justify-between">
          <div class="flex-1">
            <h3 class="font-semibold text-lg text-gray-900 dark:text-gray-100">
              {{ run.name }}
            </h3>
            <p class="text-sm text-gray-500 dark:text-gray-400 font-mono mt-1">
              {{ run.run_id }}
            </p>
          </div>
          <div v-if="run.private !== undefined" class="ml-4">
            <span
              class="px-2 py-1 text-xs rounded font-medium"
              :class="{
                'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400':
                  run.private,
                'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400':
                  !run.private,
              }"
            >
              {{ run.private ? "Private" : "Public" }}
            </span>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 text-sm">
          <div>
            <div class="text-gray-500 dark:text-gray-500">Created</div>
            <div class="text-gray-900 dark:text-gray-100">
              {{ formatDate(run.created_at) }}
            </div>
          </div>
          <div v-if="run.last_synced_at">
            <div class="text-gray-500 dark:text-gray-500">Last Synced</div>
            <div class="text-gray-900 dark:text-gray-100">
              {{ formatDate(run.last_synced_at) }}
            </div>
          </div>
          <div v-if="run.total_size">
            <div class="text-gray-500 dark:text-gray-500">Size</div>
            <div class="text-gray-900 dark:text-gray-100">
              {{ formatSize(run.total_size) }}
            </div>
          </div>
          <div v-if="run.config && Object.keys(run.config).length > 0">
            <div class="text-gray-500 dark:text-gray-500">Config</div>
            <div class="text-gray-900 dark:text-gray-100 text-xs">
              {{ Object.keys(run.config).length }} params
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
