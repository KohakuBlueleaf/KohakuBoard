<template>
  <nav
    class="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-6 py-3 shadow-sm"
  >
    <div class="flex items-center justify-between max-w-full mx-auto">
      <div class="flex items-center gap-3">
        <!-- Sidebar toggle button (project page on mobile) -->
        <button
          v-if="projectSidebarState"
          @click="handleSidebarToggle"
          class="p-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors active:bg-gray-200 dark:active:bg-gray-700"
          :title="
            projectSidebarState.collapsed
              ? 'Show runs sidebar'
              : 'Hide runs sidebar'
          "
        >
          <svg
            v-if="projectSidebarState.collapsed"
            class="w-6 h-6 text-gray-700 dark:text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
          <svg
            v-else
            class="w-6 h-6 text-gray-700 dark:text-gray-300"
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

        <router-link to="/" class="flex items-center gap-2">
          <img
            src="/images/logo-square.svg"
            alt="KohakuBoard"
            class="h-8 w-8"
          />
          <h1
            class="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent"
          >
            KohakuBoard
          </h1>
        </router-link>
        <!-- Dynamic breadcrumb navigation -->
        <el-breadcrumb separator="/" class="text-sm">
          <el-breadcrumb-item :to="{ path: '/projects' }"
            >Projects</el-breadcrumb-item
          >
          <el-breadcrumb-item
            v-if="currentProject"
            :to="{ path: `/projects/${currentProject}` }"
          >
            {{ currentProject }}
          </el-breadcrumb-item>
          <el-breadcrumb-item v-if="currentRun">
            {{ currentRun }}
          </el-breadcrumb-item>
        </el-breadcrumb>
      </div>
      <div class="flex items-center gap-2">
        <!-- Auth UI (only in remote mode) -->
        <template v-if="isRemoteMode">
          <template v-if="authStore.isAuthenticated">
            <!-- User dropdown -->
            <el-dropdown>
              <div
                class="flex items-center gap-2 cursor-pointer px-3 py-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <div
                  class="i-ep-user text-xl text-gray-700 dark:text-gray-300"
                />
                <span class="text-sm text-gray-900 dark:text-gray-100">
                  {{ authStore.username }}
                </span>
                <div class="i-ep-arrow-down text-gray-500 dark:text-gray-400" />
              </div>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item @click="handleLogout">
                    <div class="i-ep-switch-button inline-block mr-2" />
                    Logout
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </template>
          <template v-else>
            <el-button @click="$router.push('/login')" size="small" plain>
              Login
            </el-button>
            <el-button
              type="primary"
              @click="$router.push('/register')"
              size="small"
            >
              Sign Up
            </el-button>
          </template>
        </template>

        <!-- Theme toggle -->
        <button
          @click="toggleAnimations"
          class="p-2 rounded-md transition-colors bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-lg"
          :title="
            animationsEnabled ? 'Disable animations' : 'Enable animations'
          "
        >
          <span v-if="animationsEnabled">🎬</span>
          <span v-else>⏸️</span>
        </button>
        <button
          @click="toggleDarkMode"
          class="p-2 rounded-md transition-colors bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-lg"
          title="Toggle dark mode"
        >
          <span v-if="darkMode">☀️</span>
          <span v-else>🌙</span>
        </button>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { useAnimationPreference } from "@/composables/useAnimationPreference";
import { useAuthStore } from "@/stores/auth";
import { getSystemInfo } from "@/utils/api";
import { ElMessage } from "element-plus";

const { animationsEnabled, toggleAnimations } = useAnimationPreference();
const authStore = useAuthStore();
const router = useRouter();
const route = useRoute();
const systemInfo = ref(null);

// Extract current project and run from route for breadcrumb
const currentProject = computed(() => route.params.project || null);
const currentRun = computed(() => route.params.id || null);

// Check for project page sidebar state (provided via window object)
const projectSidebarState = ref(null);

// Poll for sidebar state updates
watch(
  () => route.path,
  () => {
    projectSidebarState.value = window.__projectSidebarState || null;
  },
  { immediate: true },
);

// Also check periodically in case state updates
let sidebarStateInterval = null;
onMounted(() => {
  sidebarStateInterval = setInterval(() => {
    projectSidebarState.value = window.__projectSidebarState || null;
  }, 100);
});

onUnmounted(() => {
  if (sidebarStateInterval) {
    clearInterval(sidebarStateInterval);
  }
});

function handleSidebarToggle() {
  if (projectSidebarState.value?.toggle) {
    projectSidebarState.value.toggle();
  }
}

const props = defineProps({
  darkMode: {
    type: Boolean,
    required: true,
  },
  showSidebarToggle: {
    type: Boolean,
    default: false,
  },
  sidebarCollapsed: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(["toggle-dark-mode", "toggle-sidebar"]);

onMounted(async () => {
  try {
    systemInfo.value = await getSystemInfo();
  } catch (err) {
    console.error("Failed to load system info:", err);
  }
});

const isRemoteMode = computed(() => systemInfo.value?.mode === "remote");

function toggleDarkMode() {
  emit("toggle-dark-mode");
}

async function handleLogout() {
  try {
    await authStore.logout();
    ElMessage.success("Logged out successfully");
    router.push("/");
  } catch (err) {
    ElMessage.error("Logout failed");
  }
}
</script>
