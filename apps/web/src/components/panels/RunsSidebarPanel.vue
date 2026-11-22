<script setup>
import { computed } from "vue"
import { useProjectsStore } from "@/stores/projects"
import { useTabsStore } from "@/stores/tabs"
import { useWorkspaceStore } from "@/stores/workspace"
import { statusColor } from "@/utils/palette"
import { formatCount, formatRelativeTime } from "@/utils/format"

const props = defineProps({
  scopeId: { type: String, required: true },
  project: { type: String, required: true },
  lockedRunId: { type: String, default: null },
})

const projects = useProjectsStore()
const tabs = useTabsStore()
const workspace = useWorkspaceStore()

const scope = computed(() => workspace.getScope(props.scopeId))
const allRuns = computed(() => projects.getRuns(props.project))
const filtered = computed(() => workspace.filteredRuns(props.scopeId, allRuns.value, props.lockedRunId))

const states = [
  { id: "all", label: "All" },
  { id: "running", label: "Live" },
  { id: "finished", label: "Done" },
  { id: "crashed", label: "Err" },
]

function showAll() {
  workspace.setAllRunsVisibility(
    props.scopeId,
    filtered.value.map((r) => r.run_id),
    true,
  )
}

function hideAll() {
  workspace.setAllRunsVisibility(
    props.scopeId,
    filtered.value.map((r) => r.run_id),
    false,
  )
}

function onColorPick(runId, ev) {
  ev.stopPropagation()
  workspace.setRunColor(runId, ev.target.value)
}
</script>

<template>
  <div class="h-full flex flex-col kt-v2-panel overflow-hidden" data-testid="runs-sidebar">
    <div class="p-2.5 border-b kt-v2-line flex flex-col gap-2 shrink-0">
      <div class="flex items-center justify-between">
        <span class="text-xs font-semibold uppercase tracking-wide text-[var(--v2-ink-2)]">
          Runs ({{ filtered.length }})
        </span>
        <div class="flex items-center gap-1">
          <button
            class="btn-ghost text-[10px] px-1.5"
            title="Toggle legend labels between Run Name and Run ID"
            @click="scope.legendMode = scope.legendMode === 'name' ? 'run_id' : 'name'"
          >
            {{ scope.legendMode === "name" ? "Name" : "ID" }}
          </button>
          <button class="btn-ghost text-[11px] px-1.5" @click="showAll">All</button>
          <button class="btn-ghost text-[11px] px-1.5" @click="hideAll">None</button>
        </div>
      </div>

      <input
        v-model="scope.runSearch"
        type="text"
        placeholder="Filter runs, groups, tags..."
        class="w-full px-2.5 py-1.5 text-xs rounded-lg border kt-v2-line bg-[var(--v2-card)] text-[var(--v2-ink)] outline-none focus:border-iolite"
      />

      <div class="flex items-center gap-1">
        <button
          v-for="st in states"
          :key="st.id"
          class="flex-1 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer"
          :class="
            scope.stateFilter === st.id
              ? 'bg-iolite text-white'
              : 'bg-[var(--v2-card)] text-[var(--v2-ink-2)] hover:text-[var(--v2-ink)]'
          "
          @click="scope.stateFilter = st.id"
        >
          {{ st.label }}
        </button>
      </div>
    </div>

    <div class="flex-1 overflow-y-auto py-1">
      <div
        v-for="run in filtered"
        :key="run.run_id"
        class="kt-row px-2.5 py-1.5 flex items-center gap-2 border-b kt-v2-line/50 cursor-pointer"
        :class="{ 'opacity-45': !workspace.isRunVisible(scopeId, run.run_id) }"
      >
        <!-- Custom color dot (click opens native color picker!) -->
        <label
          class="relative w-3.5 h-3.5 shrink-0 flex items-center justify-center cursor-pointer"
          title="Click to customize run color"
          @click.stop
        >
          <input
            type="color"
            :value="workspace.runColor(run.run_id, allRuns)"
            class="absolute inset-0 opacity-0 w-full h-full cursor-pointer"
            @input="onColorPick(run.run_id, $event)"
          />
          <span
            class="w-2.5 h-2.5 rounded-full block border border-black/15"
            :style="{ background: workspace.runColor(run.run_id, allRuns) }"
          />
        </label>

        <!-- Click run row to open first-class Run View tab -->
        <div
          class="min-w-0 flex-1"
          :title="`Open Run View: ${run.name}`"
          @click="tabs.openRunTab(project, run.run_id, run.name)"
        >
          <div class="flex items-center gap-1.5">
            <span
              class="w-1.5 h-1.5 rounded-full shrink-0"
              :style="{ background: statusColor(run.state) }"
              :title="run.state"
            />
            <span class="text-xs font-medium truncate hover:text-iolite">{{ run.name }}</span>
          </div>
          <div class="flex items-center justify-between text-[10px] text-[var(--v2-ink-3)] mt-0.5">
            <span class="num">{{ run.run_id }} · {{ formatCount(run.last_step + 1) }} st</span>
            <span>{{ formatRelativeTime(run.updated_at_ms) }}</span>
          </div>
        </div>

        <!-- Eye toggle visibility in project comparison charts -->
        <button
          class="w-5 h-5 rounded flex items-center justify-center shrink-0 text-[var(--v2-ink-2)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-tab-hover)] cursor-pointer"
          :title="workspace.isRunVisible(scopeId, run.run_id) ? 'Hide from comparison' : 'Show in comparison'"
          @click.stop="workspace.toggleRunVisibility(scopeId, run.run_id)"
        >
          <span
            :class="workspace.isRunVisible(scopeId, run.run_id) ? 'i-carbon-view' : 'i-carbon-view-off'"
            class="text-xs"
          />
        </button>
      </div>

      <div v-if="filtered.length === 0" class="p-4 text-center text-xs text-[var(--v2-ink-3)]">
        No matching runs
      </div>
    </div>
  </div>
</template>
