<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 py-12 px-4">
    <article class="max-w-4xl mx-auto">
      <div
        class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 shadow-sm"
      >
        <MarkdownViewer :content="content" :strip-frontmatter="true" />
      </div>
    </article>
  </div>
</template>

<script setup>
import MarkdownViewer from "@/components/common/MarkdownViewer.vue";

const props = defineProps({
  content: {
    type: String,
    required: true,
  },
});
</script>

<style scoped>
/* Additional styling for markdown pages */
.markdown-body {
  max-width: none;
}
</style>
