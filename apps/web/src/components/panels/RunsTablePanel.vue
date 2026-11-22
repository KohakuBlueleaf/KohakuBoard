<script setup>
import { computed, ref } from "vue"
import { useProjectsStore } from "@/stores/projects"
import { useTabsStore } from "@/stores/tabs"
import { useWorkspaceStore } from "@/stores/workspace"
import { statusColor } from "@/utils/palette"
import { formatCount, formatDuration, formatNumber } from "@/utils/format"

const props = defineProps({
  scopeId: { type: String, required: true },
  project: { type: String, required: true },
  lockedRunId: { type: String, default: null },
})

const projects = useProjectsStore()
const tabs = useTabsStore()
const workspace = useWorkspaceStore()

const sortKey = ref("updated_at_ms")
const sortDesc = ref(true)

const scope = computed(() => workspace.getScope(props.scopeId))
const allRuns = computed(() => projects.getRuns(props.project))
const filtered = computed(() => workspace.filteredRuns(props.scopeId, allRuns.value, props.lockedRunId))

const configKeys = computed(() => {
  const keys = new Set()
  for (const r of filtered.value) {
    for (const k of Object.keys(r.config || {})) keys.add(k)
  }
  const list = Array.from(keys).sort()
  if (!scope.value.diffOnlyConfig || filtered.value.length <= 1) return list.slice(0, 8)
  return list
    .filter((k) => {
      const vals = new Set(filtered.value.map((r) => JSON.stringify((r.config || {})[k])))
      return vals.size > 1
    })
    .slice(0, 8)
})

const summaryKeys = computed(() => {
  const keys = new Set()
  for (const r of filtered.value) {
    for (const k of Object.keys(r.summary || {})) keys.add(k)
  }
  return Array.from(keys).sort().slice(0, 8)
})

const sortedRuns = computed(() => {
  const list = [...filtered.value]
  list.sort((a, b) => {
    const av = extractVal(a, sortKey.value)
    const bv = extractVal(b, sortKey.value)
    if (av === bv) return 0
    if (av === null || av === undefined) return 1
    if (bv === null || bv === undefined) return -1
    const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv))
    return sortDesc.value ? -cmp : cmp
  })
  return list
})

function extractVal(run, key) {
  if (key.startsWith("config.")) return (run.config || {})[key.slice(7)]
  if (key.startsWith("summary.")) return (run.summary || {})[key.slice(8)]?.last
  return run[key]
}

function toggleSort(key) {
  if (sortKey.value === key) sortDesc.value = !sortDesc.value
  else {
    sortKey.value = key
    sortDesc.value = true
  }
}

function fmtCell(val) {
  if (val === null || val === undefined) return "—"
  if (typeof val === "number") return formatNumber(val, 4)
  if (typeof val === "object") return JSON.stringify(val)
  return String(val)
}
</script>

<template>
  <div class="h-full flex flex-col kt-v2-canvas overflow-hidden" data-testid="runs-table">
    <div class="px-3 py-2 border-b kt-v2-line flex items-center justify-between gap-2 shrink-0">
      <div class="flex items-center gap-2">
        <span class="text-xs font-semibold">Run Comparison Table</span>
        <span class="text-[11px] num text-[var(--v2-ink-3)]">{{ sortedRuns.length }} runs</span>
      </div>
      <label class="flex items-center gap-1.5 text-xs text-[var(--v2-ink-2)] cursor-pointer">
        <input v-model="scope.diffOnlyConfig" type="checkbox" class="accent-iolite" />
        Diff-only config columns
      </label>
    </div>

    <div class="flex-1 overflow-auto">
      <table class="w-full text-left border-collapse text-xs">
        <thead class="sticky top-0 kt-v2-panel border-b kt-v2-edge z-10">
          <tr>
            <th class="px-2.5 py-2 font-medium cursor-pointer" @click="toggleSort('name')">Run</th>
            <th class="px-2 py-2 font-medium cursor-pointer" @click="toggleSort('state')">State</th>
            <th class="px-2 py-2 font-medium cursor-pointer" @click="toggleSort('last_step')">Steps</th>
            <th class="px-2 py-2 font-medium">Duration</th>
            <th
              v-for="ck in configKeys"
              :key="'c-' + ck"
              class="px-2 py-2 font-medium text-sapphire dark:text-sapphire-light cursor-pointer whitespace-nowrap"
              @click="toggleSort('config.' + ck)"
            >
              cfg.{{ ck }}
            </th>
            <th
              v-for="sk in summaryKeys"
              :key="'s-' + sk"
              class="px-2 py-2 font-medium text-aquamarine-shadow dark:text-aquamarine-light cursor-pointer whitespace-nowrap"
              @click="toggleSort('summary.' + sk)"
            >
              {{ sk }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="run in sortedRuns"
            :key="run.run_id"
            class="kt-row border-b kt-v2-line cursor-pointer"
            @click="tabs.openRunTab(project, run.run_id, run.name)"
          >
            <td class="px-2.5 py-1.5 whitespace-nowrap">
              <div class="flex items-center gap-2">
                <span
                  class="w-2.5 h-2.5 rounded-full shrink-0"
                  :style="{ background: workspace.runColor(run.run_id, allRuns) }"
                />
                <span class="font-medium hover:text-iolite">{{ run.name }}</span>
                <span class="num text-[10px] text-[var(--v2-ink-3)]">{{ run.run_id }}</span>
              </div>
            </td>
            <td class="px-2 py-1.5 whitespace-nowrap">
              <span class="inline-flex items-center gap-1 text-[11px]">
                <span class="w-1.5 h-1.5 rounded-full" :style="{ background: statusColor(run.state) }" />
                {{ run.state }}
              </span>
            </td>
            <td class="px-2 py-1.5 num">{{ formatCount(run.last_step + 1) }}</td>
            <td class="px-2 py-1.5 num text-[var(--v2-ink-2)]">
              {{ formatDuration(run.created_at_ms, run.finished_at_ms) }}
            </td>
            <td v-for="ck in configKeys" :key="run.run_id + '-c-' + ck" class="px-2 py-1.5 num">
              {{ fmtCell((run.config || {})[ck]) }}
            </td>
            <td v-for="sk in summaryKeys" :key="run.run_id + '-s-' + sk" class="px-2 py-1.5 num">
              {{ fmtCell((run.summary || {})[sk]?.last) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
