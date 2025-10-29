<script setup>
import { ref, onMounted } from "vue";
import { initializeSliderSync } from "@/composables/useSliderSync";
import { useAuthStore } from "@/stores/auth";
import { getSystemInfo } from "@/utils/api";
import TheHeader from "@/components/layout/TheHeader.vue";
import TheFooter from "@/components/layout/TheFooter.vue";

const darkMode = ref(false);
const systemInfo = ref(null);
const authStore = useAuthStore();

onMounted(async () => {
  const savedTheme = localStorage.getItem("theme");
  darkMode.value =
    savedTheme === "dark" ||
    (!savedTheme && window.matchMedia("(prefers-color-scheme: dark)").matches);
  updateTheme();

  // Initialize global slider synchronization
  initializeSliderSync();

  // Fetch system info for mode detection
  try {
    systemInfo.value = await getSystemInfo();

    // Initialize auth store if in remote mode
    if (systemInfo.value?.mode === "remote") {
      await authStore.init();
    }
  } catch (error) {
    console.error("Failed to fetch system info:", error);
  }
});

function toggleDarkMode() {
  darkMode.value = !darkMode.value;
  updateTheme();
}

function updateTheme() {
  if (darkMode.value) {
    document.documentElement.classList.add("dark");
    localStorage.setItem("theme", "dark");
  } else {
    document.documentElement.classList.remove("dark");
    localStorage.setItem("theme", "light");
  }
}
</script>

<template>
  <div
    class="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors flex flex-col"
  >
    <TheHeader :dark-mode="darkMode" @toggle-dark-mode="toggleDarkMode" />
    <main class="flex-1 max-w-full mx-auto px-6 py-6 w-full">
      <router-view />
    </main>
    <TheFooter />
  </div>
</template>

<style>
/* Global animation control */
.no-animations,
.no-animations *,
.no-animations *::before,
.no-animations *::after {
  animation-duration: 0.01ms !important;
  animation-iteration-count: 1 !important;
  transition-duration: 0.01ms !important;
  transition-delay: 0ms !important;
}

/* Also respect system preference */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    transition-delay: 0ms !important;
  }
}
</style>
