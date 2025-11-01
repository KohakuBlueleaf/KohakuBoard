<script setup>
import { ref, computed, onMounted } from "vue";
import { useRouter } from "vue-router";
import { useAuthStore } from "@/stores/auth";
import { fetchProjects, getSystemInfo } from "@/utils/api";

const router = useRouter();
const authStore = useAuthStore();
const projects = ref([]);
const loading = ref(true);
const systemInfo = ref(null);

onMounted(async () => {
  try {
    // Get system mode
    systemInfo.value = await getSystemInfo();

    // Check if authentication required
    if (systemInfo.value?.require_auth && !authStore.isAuthenticated) {
      // Don't fetch - show login prompt instead
      loading.value = false;
      return;
    }

    // Fetch projects
    const result = await fetchProjects();
    projects.value = result.projects;
  } catch (error) {
    console.error("Failed to fetch projects:", error);
  } finally {
    loading.value = false;
  }
});

const isRemoteMode = computed(() => systemInfo.value?.mode === "remote");
const needsAuth = computed(
  () => systemInfo.value?.require_auth && !authStore.isAuthenticated,
);

function viewProject(projectName) {
  router.push(`/projects/${projectName}`);
}

function formatDate(timestamp) {
  if (!timestamp) return "N/A";
  return new Date(timestamp).toLocaleString();
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-gray-100">
        Projects
      </h1>
    </div>

    <!-- Auth required prompt (remote mode only) -->
    <div
      v-if="needsAuth"
      class="bg-white dark:bg-gray-900 rounded-lg shadow-md p-8 text-center border border-gray-200 dark:border-gray-800"
    >
      <div
        class="i-ep-lock text-6xl text-gray-400 dark:text-gray-600 mb-4 inline-block"
      />
      <h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
        Authentication Required
      </h2>
      <p class="text-gray-600 dark:text-gray-400 mb-6">
        Please login to view your projects
      </p>
      <div class="flex gap-3 justify-center">
        <el-button @click="$router.push('/login')" plain size="large">
          Login
        </el-button>
        <el-button
          type="primary"
          @click="$router.push('/register')"
          size="large"
        >
          Sign Up
        </el-button>
      </div>
    </div>

    <div v-else-if="loading" class="text-center py-12">
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
