<script setup>
import { ref, onMounted } from "vue";
import { fetchProjects } from "@/utils/api";
import { useRouter } from "vue-router";

const router = useRouter();
const projects = ref([]);
const loading = ref(true);

onMounted(async () => {
  try {
    const result = await fetchProjects();
    projects.value = result.projects;
  } catch (error) {
    console.error("Failed to fetch projects:", error);
  } finally {
    loading.value = false;
  }
});

function viewProject(projectName) {
  router.push(`/projects/${projectName}`);
}

function formatDate(timestamp) {
  if (!timestamp) return "N/A";
  return new Date(timestamp).toLocaleString();
}

function formatSteps(steps) {
  if (steps === 0 || steps === undefined) return "N/A";
  return steps.toLocaleString();
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-gray-100">
        Dashboard
      </h1>
      <router-link
        to="/experiments"
        class="px-4 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white rounded-md font-medium transition-colors shadow-sm"
      >
        View All Experiments
      </router-link>
    </div>

    <div v-if="loading" class="text-center py-12">
      <div class="text-gray-500 dark:text-gray-400">Loading projects...</div>
    </div>

    <div
      v-else-if="projects.length === 0"
      class="bg-white dark:bg-gray-900 rounded-lg shadow-md p-8 text-center border border-gray-200 dark:border-gray-800"
    >
      <div class="text-gray-500 dark:text-gray-400 mb-4">No projects found</div>
      <p class="text-sm text-gray-400 dark:text-gray-500">
        Start tracking your ML experiments with KohakuBoard client library.
      </p>
    </div>

    <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <div
        v-for="project in projects"
        :key="project.name"
        @click="viewProject(project.name)"
        class="bg-white dark:bg-gray-900 rounded-lg shadow-md p-6 cursor-pointer hover:shadow-lg transition-all border border-gray-200 dark:border-gray-800 hover:border-blue-400 dark:hover:border-blue-600"
      >
        <h3 class="font-semibold text-xl text-gray-900 dark:text-gray-100 mb-2">
          {{ project.display_name }}
        </h3>
        <div class="text-gray-600 dark:text-gray-400 mb-4">
          {{ project.run_count }} {{ project.run_count === 1 ? "run" : "runs" }}
        </div>
        <div
          v-if="project.updated_at"
          class="text-xs text-gray-500 dark:text-gray-500"
        >
          Updated: {{ formatDate(project.updated_at) }}
        </div>
      </div>
    </div>
  </div>
</template>
