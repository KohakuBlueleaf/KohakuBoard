<script setup>
import { computed, ref, watch } from "vue"
import { useProjectsStore } from "@/stores/projects"
import { useWorkspaceStore } from "@/stores/workspace"
import { api } from "@/utils/api"

const props = defineProps({
  scopeId: { type: String, required: true },
  project: { type: String, required: true },
  lockedRunId: { type: String, default: null },
})

const projects = useProjectsStore()
const workspace = useWorkspaceStore()

const searchQuery = ref("")
const streamFilter = ref("all")
const lines = ref([])
const loading = ref(false)

const allRuns = computed(() => projects.getRuns(props.project))
const targetRun = computed(() => {
  if (props.lockedRunId) return allRuns.value.find((r) => r.run_id === props.lockedRunId) || null
  const vis = workspace.visibleRuns(props.scopeId, allRuns.value)
  return vis[0] || allRuns.value[0] || null
})

async function loadLogs() {
  if (!targetRun.value) {
    lines.value = []
    return
  }
  loading.value = true
  try {
    const q = searchQuery.value.trim()
    if (q) {
      lines.value = await api.logsSearch(props.project, targetRun.value.run_id, q, 100)
    } else {
      const tail = await api.logsTail(props.project, targetRun.value.run_id, { limit: 250 })
      lines.value = tail.reverse()
    }
  } finally {
    loading.value = false
  }
}

const filteredLines = computed(() =>
  streamFilter.value === "all"
    ? lines.value
    : lines.value.filter((l) => l.stream === streamFilter.value),
)

watch([targetRun, searchQuery], loadLogs, { immediate: true })
</script>

<template>
  <div class="h-full flex flex-col kt-v2-canvas overflow-hidden" data-testid="console-logs">
    <div class="px-3 py-2 border-b kt-v2-line flex items-center gap-2 shrink-0">
      <span class="text-xs font-semibold shrink-0">Console Logs</span>
      <input
        v-model="searchQuery"
        type="text"
        placeholder="FTS5 BM25 search (e.g. NaN, warning)..."
        class="flex-1 px-2.5 py-1 text-xs rounded-lg border kt-v2-line bg-[var(--v2-card)] text-[var(--v2-ink)] outline-none focus:border-iolite"
      />
      <div class="flex items-center gap-1">
        <button
          v-for="s in ['all', 'stdout', 'stderr']"
          :key="s"
          class="px-2 py-0.5 rounded text-[11px] cursor-pointer"
          :class="
            streamFilter === s
              ? 'bg-iolite text-white'
              : 'bg-[var(--v2-panel)] text-[var(--v2-ink-2)]'
          "
          @click="streamFilter = s"
        >
          {{ s }}
        </button>
      </div>
    </div>

    <div class="flex-1 overflow-auto p-2.5 font-mono text-[11.5px] leading-relaxed">
      <div
        v-for="entry in filteredLines"
        :key="entry.id"
        class="flex items-start gap-2 py-0.5 border-b kt-v2-line/40"
      >
        <span
          class="px-1 rounded text-[10px] shrink-0"
          :class="
            entry.stream === 'stderr'
              ? 'bg-coral-light text-coral-shadow dark:bg-coral-shadow/40 dark:text-coral-light'
              : 'bg-[var(--v2-panel)] text-[var(--v2-ink-3)]'
          "
        >
          {{ entry.stream }}
        </span>
        <span class="num text-[10px] text-[var(--v2-ink-3)] shrink-0">st={{ entry.step }}</span>
        <span class="whitespace-pre-wrap break-words">{{ entry.text }}</span>
      </div>
      <div
        v-if="filteredLines.length === 0"
        class="h-24 flex items-center justify-center text-xs text-[var(--v2-ink-3)] font-sans"
      >
        {{ loading ? "Searching..." : "No log lines." }}
      </div>
    </div>
  </div>
</template>
