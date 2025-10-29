<script setup>
import { ref, onMounted, computed, watch } from "vue";
import { useRoute } from "vue-router";
import MarkdownPage from "@/components/common/MarkdownPage.vue";

const route = useRoute();
const markdownContent = ref("");
const loading = ref(true);
const error = ref(null);

const slug = computed(() => route.params.slug);
const docTitle = computed(() => {
  const titles = {
    api: "API Reference",
    cli: "CLI Tools",
    "getting-started": "Getting Started",
  };
  return titles[slug.value] || slug.value;
});

async function loadDoc() {
  loading.value = true;
  error.value = null;

  try {
    const response = await fetch(`/docs/${slug.value}.md`);
    if (!response.ok) {
      throw new Error(`Documentation not found: ${slug.value}`);
    }
    const text = await response.text();
    markdownContent.value = text;
  } catch (err) {
    error.value = err.message;
    console.error("Failed to load documentation:", err);
  } finally {
    loading.value = false;
  }
}

// Load on mount and when slug changes
onMounted(loadDoc);
watch(slug, loadDoc);
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 py-12 px-4">
    <div class="max-w-4xl mx-auto">
      <!-- Back button -->
      <div class="mb-6">
        <router-link
          to="/docs"
          class="inline-flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
        >
          <span class="i-ep-ArrowLeft" />
          Back to Documentation
        </router-link>
      </div>

      <!-- Loading state -->
      <div v-if="loading" class="text-center py-12">
        <div
          class="i-ep-Loading animate-spin text-4xl text-blue-600 dark:text-blue-400 mx-auto mb-4"
        />
        <p class="text-gray-600 dark:text-gray-400">
          Loading {{ docTitle }}...
        </p>
      </div>

      <!-- Error state -->
      <div
        v-else-if="error"
        class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6"
      >
        <h2 class="text-xl font-bold text-red-900 dark:text-red-100 mb-2">
          Documentation Not Found
        </h2>
        <p class="text-red-700 dark:text-red-300">{{ error }}</p>
        <p class="text-sm text-red-600 dark:text-red-400 mt-4">
          Make sure you ran the build script to copy documentation files.
        </p>
        <div class="mt-4">
          <router-link
            to="/docs"
            class="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <span class="i-ep-ArrowLeft" />
            Back to Docs
          </router-link>
        </div>
      </div>

      <!-- Content -->
      <div v-else>
        <MarkdownPage :content="markdownContent" />
      </div>

      <!-- Navigation (only show if content loaded successfully) -->
      <div
        v-if="!loading && !error"
        class="mt-8 flex justify-between max-w-4xl mx-auto"
      >
        <router-link
          to="/docs"
          class="inline-flex items-center gap-2 px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-lg transition-colors"
        >
          <span class="i-ep-ArrowLeft" />
          Documentation Home
        </router-link>
      </div>
    </div>
  </div>
</template>
