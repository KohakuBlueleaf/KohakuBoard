<template>
  <div
    ref="markdownContainer"
    class="markdown-body markdown-container"
    v-html="renderedHTML"
  />
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from "vue";
import { renderMarkdown } from "@/utils/markdown";
import mermaid from "mermaid";
import Panzoom from "panzoom";

const props = defineProps({
  content: {
    type: String,
    required: true,
  },
  stripFrontmatter: {
    type: Boolean,
    default: true,
  },
});

const markdownContainer = ref(null);

const renderedHTML = computed(() => {
  return renderMarkdown(props.content, {
    stripFrontmatter: props.stripFrontmatter,
  });
});

// Check if dark mode
const isDark = computed(() => {
  if (typeof document !== "undefined") {
    return document.documentElement.classList.contains("dark");
  }
  return false;
});

// Mermaid configuration
const getMermaidConfig = (dark) => ({
  startOnLoad: false,
  securityLevel: "loose",
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
  theme: dark ? "dark" : "default",
  logLevel: "fatal",
});

// Initialize Mermaid
mermaid.initialize(getMermaidConfig(isDark.value));

// Render mermaid diagrams
async function renderMermaidDiagrams() {
  await nextTick();

  if (!markdownContainer.value) return;

  mermaid.initialize(getMermaidConfig(isDark.value));

  const mermaidBlocks = markdownContainer.value.querySelectorAll(
    "pre code.language-mermaid",
  );
  const existingWrappers =
    markdownContainer.value.querySelectorAll(".mermaid-wrapper");

  if (mermaidBlocks.length > 0) {
    // Initial render
    for (let i = 0; i < mermaidBlocks.length; i++) {
      const block = mermaidBlocks[i];
      const code = block.textContent;
      const pre = block.parentElement;

      try {
        const wrapper = document.createElement("div");
        wrapper.className = "mermaid-wrapper";
        wrapper.setAttribute("data-mermaid-code", code);

        pre.replaceWith(wrapper);
        await renderSingleDiagram(wrapper, code, i);
      } catch (err) {
        console.error("Mermaid rendering error:", err);
      }
    }
  } else if (existingWrappers.length > 0) {
    // Theme change: re-render
    for (let i = 0; i < existingWrappers.length; i++) {
      const wrapper = existingWrappers[i];
      const code = wrapper.getAttribute("data-mermaid-code");
      if (code) {
        wrapper
          .querySelectorAll(".mermaid-diagram, .mermaid-controls")
          .forEach((el) => el.remove());
        await renderSingleDiagram(wrapper, code, i);
      }
    }
  }
}

// Render single diagram
async function renderSingleDiagram(wrapper, code, index) {
  const id = `mermaid-${Date.now()}-${index}`;
  const container = document.createElement("div");
  container.className = "mermaid-diagram";
  container.id = id;

  let svg;
  try {
    const result = await mermaid.render(id, code);
    svg = result.svg;
  } catch (err) {
    console.error("Mermaid render error:", err);
    container.innerHTML = `
      <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded p-4 text-sm">
        <div class="font-semibold text-red-600 dark:text-red-400 mb-2">Diagram Syntax Error</div>
        <div class="text-red-700 dark:text-red-300 text-xs">${err.message}</div>
      </div>
    `;
    wrapper.appendChild(container);
    return;
  }

  container.innerHTML = svg;

  const svgElement = container.querySelector("svg");
  if (svgElement) {
    svgElement.style.backgroundColor = "transparent";
  }

  // Create zoom controls
  const controls = document.createElement("div");
  controls.className = "mermaid-controls";
  controls.innerHTML = `
    <button class="mermaid-zoom-btn mermaid-zoom-in" title="Zoom In">
      <div class="i-ep-ZoomIn"></div>
    </button>
    <button class="mermaid-zoom-btn mermaid-zoom-out" title="Zoom Out">
      <div class="i-ep-ZoomOut"></div>
    </button>
    <button class="mermaid-zoom-btn mermaid-zoom-reset" title="Reset Zoom">
      <div class="i-ep-RefreshLeft"></div>
    </button>
  `;

  // Initialize panzoom
  let panzoomInstance = null;
  if (svgElement) {
    panzoomInstance = Panzoom(svgElement, {
      maxZoom: 3,
      minZoom: 0.3,
      bounds: false,
      boundsPadding: 0.1,
    });

    container.addEventListener("wheel", (e) =>
      panzoomInstance.zoomWithWheel(e),
    );
  }

  // Button handlers
  controls.querySelector(".mermaid-zoom-in").addEventListener("click", (e) => {
    e.stopPropagation();
    if (panzoomInstance) panzoomInstance.zoomIn();
  });

  controls.querySelector(".mermaid-zoom-out").addEventListener("click", (e) => {
    e.stopPropagation();
    if (panzoomInstance) panzoomInstance.zoomOut();
  });

  controls
    .querySelector(".mermaid-zoom-reset")
    .addEventListener("click", (e) => {
      e.stopPropagation();
      if (panzoomInstance) {
        panzoomInstance.moveTo(0, 0);
        panzoomInstance.zoomAbs(0, 0, 1);
      }
    });

  wrapper.appendChild(controls);
  wrapper.appendChild(container);
}

// Watch content changes
watch(
  () => props.content,
  async () => {
    await nextTick();
    await renderMermaidDiagrams();
  },
  { flush: "post" },
);

// Watch theme changes
watch(isDark, () => {
  mermaid.initialize(getMermaidConfig(isDark.value));
  renderMermaidDiagrams();
});

onMounted(() => {
  renderMermaidDiagrams();
});
</script>

<style scoped>
/* GitHub-like markdown styles */
.markdown-body {
  font-size: 16px;
  line-height: 1.6;
  word-wrap: break-word;
  color: #24292f;
}

/* Dark mode text color */
.dark .markdown-body {
  color: #e6edf3;
}

/* Headers */
.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4),
.markdown-body :deep(h5),
.markdown-body :deep(h6) {
  margin-top: 24px;
  margin-bottom: 16px;
  font-weight: 600;
  line-height: 1.25;
}

.markdown-body :deep(h1:first-child),
.markdown-body :deep(h2:first-child),
.markdown-body :deep(h3:first-child) {
  margin-top: 0;
}

.markdown-body :deep(h1) {
  font-size: 2em;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 0.3em;
}

.dark .markdown-body :deep(h1),
.dark .markdown-body :deep(h2) {
  border-bottom-color: rgba(255, 255, 255, 0.1);
}

.markdown-body :deep(h2) {
  font-size: 1.5em;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 0.3em;
}

.markdown-body :deep(h3) {
  font-size: 1.25em;
}

.markdown-body :deep(h4) {
  font-size: 1em;
}

/* Paragraphs */
.markdown-body :deep(p) {
  margin-top: 0;
  margin-bottom: 16px;
}

/* Code blocks */
.markdown-body :deep(pre) {
  padding: 16px;
  overflow: auto;
  font-size: 85%;
  line-height: 1.45;
  background-color: #f6f8fa;
  border-radius: 6px;
  margin-bottom: 16px;
}

.dark .markdown-body :deep(pre) {
  background-color: rgba(0, 0, 0, 0.3);
}

/* Code inside pre blocks - MUST match pre background */
.markdown-body :deep(pre code) {
  display: inline;
  max-width: auto;
  padding: 0;
  margin: 0;
  overflow: visible;
  line-height: inherit;
  word-wrap: normal;
  background-color: transparent !important;
  border: 0;
  color: inherit;
}

/* Force transparent on ALL children of pre code (highlight.js spans) */
.markdown-body :deep(pre code *) {
  background-color: transparent !important;
}

.dark .markdown-body :deep(pre code) {
  background-color: transparent !important;
}

.dark .markdown-body :deep(pre code *) {
  background-color: transparent !important;
}

/* Inline code (NOT in pre blocks) */
.markdown-body :deep(code) {
  padding: 0.2em 0.4em;
  margin: 0;
  font-size: 85%;
  background-color: rgba(175, 184, 193, 0.2);
  border-radius: 6px;
  font-family: ui-monospace, monospace;
}

.dark .markdown-body :deep(code) {
  background-color: rgba(255, 255, 255, 0.1);
}

/* Override inline code styles when inside pre */
.markdown-body :deep(pre code) {
  padding: 0 !important;
  background-color: transparent !important;
  border-radius: 0 !important;
}

.dark .markdown-body :deep(pre code) {
  padding: 0 !important;
  background-color: transparent !important;
  border-radius: 0 !important;
}

/* Links */
.markdown-body :deep(a) {
  color: #0969da;
  text-decoration: none;
}

.dark .markdown-body :deep(a) {
  color: #58a6ff;
}

.markdown-body :deep(a:hover) {
  text-decoration: underline;
}

/* Lists */
.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  padding-left: 2em;
  margin-bottom: 16px;
}

.markdown-body :deep(li) {
  margin-bottom: 0.25em;
}

.markdown-body :deep(li > p) {
  margin-bottom: 0;
}

/* Task lists */
.markdown-body :deep(input[type="checkbox"].task-list-checkbox) {
  margin-right: 0.5em;
  vertical-align: middle;
  pointer-events: none;
}

.markdown-body :deep(li:has(> input[type="checkbox"].task-list-checkbox)) {
  list-style-type: none;
}

/* Tables */
.markdown-body :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 16px;
  overflow: auto;
  display: block;
}

.markdown-body :deep(table th),
.markdown-body :deep(table td) {
  padding: 6px 13px;
  border: 1px solid #d0d7de;
}

.dark .markdown-body :deep(table th),
.dark .markdown-body :deep(table td) {
  border-color: rgba(255, 255, 255, 0.1);
}

.markdown-body :deep(table th) {
  font-weight: 600;
  background-color: #f6f8fa;
}

.dark .markdown-body :deep(table th) {
  background-color: rgba(0, 0, 0, 0.3);
}

.markdown-body :deep(table tr:nth-child(2n)) {
  background-color: #f6f8fa;
}

.dark .markdown-body :deep(table tr:nth-child(2n)) {
  background-color: rgba(0, 0, 0, 0.1);
}

/* Blockquotes */
.markdown-body :deep(blockquote) {
  padding: 0 1em;
  color: #57606a;
  border-left: 0.25em solid #d0d7de;
  margin-bottom: 16px;
}

.dark .markdown-body :deep(blockquote) {
  color: #8b949e;
  border-left-color: rgba(255, 255, 255, 0.2);
}

/* Horizontal rules */
.markdown-body :deep(hr) {
  height: 0.25em;
  padding: 0;
  margin: 24px 0;
  background-color: #d0d7de;
  border: 0;
}

.dark .markdown-body :deep(hr) {
  background-color: rgba(255, 255, 255, 0.1);
}

/* Images */
.markdown-body :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
}

/* Ensure all content stays within bounds */
.markdown-body :deep(*) {
  max-width: 100%;
  box-sizing: border-box;
}

/* Mermaid diagram wrapper */
.markdown-body :deep(.mermaid-wrapper) {
  position: relative;
  margin: 24px 0;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px;
  background-color: #ffffff;
  overflow: hidden;
}

.dark .markdown-body :deep(.mermaid-wrapper) {
  border-color: #374151;
  background-color: #1f2937;
}

.markdown-body :deep(.mermaid-wrapper svg) {
  background: transparent !important;
}

/* Mermaid controls */
.markdown-body :deep(.mermaid-controls) {
  position: absolute;
  top: 12px;
  right: 12px;
  display: flex;
  gap: 4px;
  z-index: 10;
  background-color: rgba(255, 255, 255, 0.95);
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.dark .markdown-body :deep(.mermaid-controls) {
  background-color: rgba(31, 41, 55, 0.95);
  border-color: #4b5563;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
}

.markdown-body :deep(.mermaid-zoom-btn) {
  width: 32px;
  height: 32px;
  border: none;
  background-color: transparent;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  color: #4b5563;
  transition: all 0.2s;
}

.markdown-body :deep(.mermaid-zoom-btn:hover) {
  background-color: rgba(59, 130, 246, 0.15);
  color: #3b82f6;
}

.dark .markdown-body :deep(.mermaid-zoom-btn) {
  color: #d1d5db;
}

.dark .markdown-body :deep(.mermaid-zoom-btn:hover) {
  background-color: rgba(59, 130, 246, 0.3);
  color: #60a5fa;
}

.markdown-body :deep(.mermaid-zoom-btn div) {
  width: 20px;
  height: 20px;
}

/* Mermaid diagram container */
.markdown-body :deep(.mermaid-diagram) {
  text-align: center;
  background: transparent;
  overflow: visible;
  min-height: 200px;
  max-height: 600px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  cursor: grab;
}

.markdown-body :deep(.mermaid-diagram:active) {
  cursor: grabbing;
}

.markdown-body :deep(.mermaid-diagram svg) {
  max-width: 100%;
  max-height: 600px;
  width: auto;
  height: auto;
}
</style>
