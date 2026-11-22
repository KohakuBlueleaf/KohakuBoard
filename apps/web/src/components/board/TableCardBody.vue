<script setup>
import { computed, ref, watch } from "vue"
import { api } from "@/utils/api"
import { useSliderSync } from "@/composables/useSliderSync"

const props = defineProps({
  project: { type: String, required: true },
  card: { type: Object, required: true },
  runs: { type: Array, required: true },
  height: { type: Number, default: 280 },
})

const emit = defineEmits(["patch-card"])

const loading = ref(false)
const activeRunId = ref("")
// Ascending sorted snapshots
const snapshots = ref([])
const search = ref("")
const lightboxUrl = ref(null)

watch(
  () => props.runs,
  (list) => {
    if (list.length === 0) activeRunId.value = ""
    else if (!list.some((r) => r.run_id === activeRunId.value)) activeRunId.value = list[0].run_id
  },
  { immediate: true },
)

async function loadTables() {
  if (!activeRunId.value || !props.card.tableName) {
    snapshots.value = []
    return
  }
  loading.value = true
  try {
    const raw = await api.tables(props.project, activeRunId.value, props.card.tableName, {
      limit: 64,
    })
    snapshots.value = [...(raw || [])].sort((a, b) => a.step - b.step)
  } catch {
    snapshots.value = []
  } finally {
    loading.value = false
  }
}

const steps = computed(() => snapshots.value.map((s) => s.step))
const autoAdvance = computed(() => props.card.autoAdvance !== false)

const stepIndex = computed({
  get() {
    if (steps.value.length === 0) return 0
    if (props.card.currentStep === null || props.card.currentStep === undefined) {
      return steps.value.length - 1
    }
    const exact = steps.value.indexOf(props.card.currentStep)
    if (exact >= 0) return exact
    if (autoAdvance.value) return steps.value.length - 1
    let best = 0
    let bestDiff = Math.abs(steps.value[0] - props.card.currentStep)
    for (let i = 1; i < steps.value.length; i++) {
      const d = Math.abs(steps.value[i] - props.card.currentStep)
      if (d < bestDiff) {
        bestDiff = d
        best = i
      }
    }
    return best
  },
  set(idx) {
    const clamped = Math.max(0, Math.min(steps.value.length - 1, idx))
    const newStep = steps.value[clamped]
    if (newStep === undefined) return
    emit("patch-card", {
      currentStep: newStep,
      autoAdvance: clamped === steps.value.length - 1,
    })
    triggerSync(newStep)
  },
})

const activeSnap = computed(() => snapshots.value[stepIndex.value] || null)

const filteredRows = computed(() => {
  if (!activeSnap.value) return []
  const q = search.value.trim().toLowerCase()
  const rows = activeSnap.value.rows || []
  if (!q) return rows.slice(0, 200)
  return rows
    .filter((row) =>
      row.some((cell) => {
        if (cell === null || cell === undefined) return false
        if (typeof cell === "object") return JSON.stringify(cell).toLowerCase().includes(q)
        return String(cell).toLowerCase().includes(q)
      }),
    )
    .slice(0, 200)
})

const { isShiftPressed, triggerSync } = useSliderSync(
  computed(() => `table:${props.card.id}`),
  steps,
  (syncedStep) => {
    emit("patch-card", { currentStep: syncedStep, autoAdvance: false })
  },
)

function isMediaCell(cell) {
  return cell && typeof cell === "object" && typeof cell.$media === "string"
}

watch([() => props.project, () => props.card.tableName, activeRunId], loadTables, {
  immediate: true,
})

defineExpose({ refresh: loadTables })
</script>

<template>
  <div class="flex flex-col h-full w-full overflow-hidden">
    <!-- Top bar: multi-run switcher + quick search -->
    <div class="px-2 py-1 border-b kt-v2-line/60 flex items-center justify-between gap-2 text-[10px] shrink-0">
      <div v-if="runs.length > 1" class="flex items-center gap-1 overflow-x-auto min-w-0">
        <button
          v-for="r in runs.slice(0, 6)"
          :key="r.run_id"
          class="px-1.5 py-0.5 rounded-full flex items-center gap-1 cursor-pointer shrink-0 border"
          :class="
            activeRunId === r.run_id
              ? 'bg-sapphire text-white border-sapphire'
              : 'bg-[var(--v2-panel)] text-[var(--v2-ink-2)] kt-v2-line'
          "
          @click="activeRunId = r.run_id"
        >
          <span class="w-1.5 h-1.5 rounded-full" :style="{ background: activeRunId === r.run_id ? '#fff' : r.color }" />
          <span class="truncate max-w-20">{{ r.label || r.name }}</span>
        </button>
      </div>
      <span v-else-if="activeSnap" class="num text-[var(--v2-ink-3)]">
        {{ activeSnap.row_count }} rows
      </span>

      <input
        v-model="search"
        type="text"
        placeholder="Filter rows…"
        class="w-28 px-1.5 py-0.5 text-[10px] rounded border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] outline-none focus:border-sapphire ml-auto"
      />
    </div>

    <!-- Table viewport -->
    <div class="flex-1 min-h-0 overflow-auto relative">
      <div
        v-if="isShiftPressed"
        class="absolute top-1.5 right-2 z-10 px-1.5 py-0.5 rounded bg-iolite text-white text-[10px] font-semibold"
      >
        SYNC
      </div>

      <table v-if="activeSnap" class="w-full text-left border-collapse text-xs">
        <thead class="kt-v2-panel sticky top-0 z-10 border-b kt-v2-edge">
          <tr>
            <th v-for="col in activeSnap.columns" :key="col" class="px-2.5 py-1.5 font-medium whitespace-nowrap">
              {{ col }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, rIdx) in filteredRows" :key="rIdx" class="kt-row border-b kt-v2-line/60">
            <td v-for="(cell, cIdx) in row" :key="cIdx" class="px-2.5 py-1 align-middle">
              <img
                v-if="isMediaCell(cell) && activeRunId"
                :src="api.blobUrl(project, activeRunId, cell.$media)"
                :alt="cell.caption || 'media'"
                class="w-10 h-10 object-contain rounded border kt-v2-line cursor-zoom-in"
                @click="lightboxUrl = api.blobUrl(project, activeRunId, cell.$media)"
              />
              <span v-else class="num">{{ cell === null ? "—" : cell }}</span>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-else class="h-full flex items-center justify-center text-xs text-[var(--v2-ink-3)]">
        {{ loading ? "Loading table…" : "No table snapshots for this run." }}
      </div>
    </div>

    <!-- Step slider footer -->
    <div
      v-if="snapshots.length > 0 && activeSnap"
      class="px-2.5 py-1 border-t kt-v2-line/70 flex items-center gap-2 text-[11px] shrink-0"
    >
      <span class="num text-[var(--v2-ink-2)] shrink-0">step {{ activeSnap.step }}</span>
      <input
        v-if="snapshots.length > 1"
        v-model.number="stepIndex"
        type="range"
        min="0"
        :max="snapshots.length - 1"
        class="flex-1 min-w-0 accent-sapphire h-1.5 cursor-pointer"
        title="Scrub table snapshot (hold Shift to sync all sliders)"
      />
      <span v-else class="flex-1" />
      <span class="num text-[10px] text-[var(--v2-ink-3)] shrink-0">
        {{ stepIndex + 1 }}/{{ snapshots.length }}
      </span>
    </div>

    <Teleport to="body">
      <div
        v-if="lightboxUrl"
        class="fixed inset-0 bg-black/75 z-50 flex items-center justify-center p-6"
        @click="lightboxUrl = null"
      >
        <img :src="lightboxUrl" class="max-h-[85vh] max-w-[90vw] object-contain rounded shadow-xl" />
      </div>
    </Teleport>
  </div>
</template>
