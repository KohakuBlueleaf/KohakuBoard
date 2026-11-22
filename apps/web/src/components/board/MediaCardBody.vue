<script setup>
import { computed, ref, watch } from "vue"
import { api } from "@/utils/api"
import { useSliderSync } from "@/composables/useSliderSync"

const props = defineProps({
  project: { type: String, required: true },
  card: { type: Object, required: true },
  runs: { type: Array, required: true }, // [{ run_id, name, label, color }]
  height: { type: Number, default: 280 },
})

const emit = defineEmits(["patch-card"])

const loading = ref(false)
// Map: run_id -> sorted ascending events [{ step, id, media_type, url, caption, mime }]
const byRun = ref({})
const lightbox = ref(null)

async function loadMedia() {
  if (!props.card.mediaName || props.runs.length === 0) {
    byRun.value = {}
    return
  }
  loading.value = true
  try {
    const next = {}
    const targets = props.runs.slice(0, 8)
    await Promise.all(
      targets.map(async (r) => {
        try {
          const evs = await api.mediaEvents(props.project, r.run_id, props.card.mediaName, {
            limit: 64,
          })
          // API returns newest-first; sort ascending for natural left=oldest -> right=newest slider
          const sorted = [...evs]
            .sort((a, b) => a.step - b.step || a.id - b.id)
            .map((ev) => ({
              ...ev,
              run_id: r.run_id,
              run_label: r.label || r.name || r.run_id,
              color: r.color,
              url: api.blobUrl(props.project, r.run_id, ev.blob_key),
            }))
          if (sorted.length > 0) next[r.run_id] = sorted
        } catch {
          // run may not have logged this media key
        }
      }),
    )
    byRun.value = next
  } finally {
    loading.value = false
  }
}

// Union of sorted ascending steps across visible runs
const steps = computed(() => {
  const s = new Set()
  for (const list of Object.values(byRun.value)) {
    for (const ev of list) s.add(ev.step)
  }
  return Array.from(s).sort((a, b) => a - b)
})

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
    // Closest step
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
    const atLatest = clamped === steps.value.length - 1
    emit("patch-card", {
      currentStep: newStep,
      autoAdvance: atLatest,
    })
    triggerSync(newStep)
  },
})

const activeStep = computed(() => steps.value[stepIndex.value] ?? null)

// Auto-advance when new steps arrive
watch(
  () => steps.value.length,
  (nextLen, prevLen) => {
    if (autoAdvance.value && nextLen > prevLen && nextLen > 0) {
      emit("patch-card", { currentStep: steps.value[nextLen - 1] })
    }
  },
)

const { isShiftPressed, triggerSync } = useSliderSync(
  computed(() => `media:${props.card.id}`),
  steps,
  (syncedStep) => {
    emit("patch-card", { currentStep: syncedStep, autoAdvance: false })
  },
)

// For each run that has media, pick the item closest to activeStep (<= activeStep preferred)
const displayedItems = computed(() => {
  if (activeStep.value === null) return []
  const target = activeStep.value
  const out = []
  for (const r of props.runs) {
    const list = byRun.value[r.run_id]
    if (!list || list.length === 0) continue
    let best = list[0]
    let bestDiff = Math.abs(list[0].step - target)
    for (const ev of list) {
      const d = Math.abs(ev.step - target)
      if (d < bestDiff || (d === bestDiff && ev.id > best.id)) {
        bestDiff = d
        best = ev
      }
    }
    out.push(best)
  }
  return out
})

watch(
  [() => props.project, () => props.card.mediaName, () => props.runs.map((r) => r.run_id).join(",")],
  loadMedia,
  { immediate: true },
)

defineExpose({ refresh: loadMedia })
</script>

<template>
  <div class="flex flex-col h-full w-full overflow-hidden">
    <!-- Media display stage -->
    <div class="flex-1 min-h-0 overflow-auto p-2 relative bg-[var(--v2-panel)]/45">
      <div
        v-if="isShiftPressed"
        class="absolute top-2 right-2 z-10 px-1.5 py-0.5 rounded bg-iolite text-white text-[10px] font-semibold"
      >
        SYNC
      </div>

      <div
        v-if="displayedItems.length === 1"
        class="h-full w-full flex flex-col items-center justify-center gap-1"
      >
        <img
          v-if="displayedItems[0].media_type === 'image'"
          :src="displayedItems[0].url"
          :alt="displayedItems[0].caption || card.mediaName"
          class="max-w-full max-h-full object-contain rounded cursor-zoom-in"
          @click="lightbox = displayedItems[0]"
        />
        <video
          v-else-if="displayedItems[0].media_type === 'video'"
          :src="displayedItems[0].url"
          controls
          class="max-w-full max-h-full rounded"
        />
        <audio
          v-else-if="displayedItems[0].media_type === 'audio'"
          :src="displayedItems[0].url"
          controls
          class="w-full max-w-xs"
        />
        <div
          v-if="displayedItems[0].caption"
          class="text-[11px] text-[var(--v2-ink-2)] truncate max-w-full px-1"
        >
          {{ displayedItems[0].caption }}
        </div>
      </div>

      <!-- Multi-run side-by-side comparison grid -->
      <div
        v-else-if="displayedItems.length > 1"
        class="grid gap-2 h-full"
        :class="displayedItems.length === 2 ? 'grid-cols-2' : 'grid-cols-2 lg:grid-cols-3'"
      >
        <div
          v-for="item in displayedItems"
          :key="`${item.run_id}-${item.id}`"
          class="kt-v2-card overflow-hidden flex flex-col cursor-zoom-in"
          @click="lightbox = item"
        >
          <div class="flex-1 min-h-20 bg-[var(--v2-panel)] flex items-center justify-center overflow-hidden p-1">
            <img
              v-if="item.media_type === 'image'"
              :src="item.url"
              :alt="item.caption || card.mediaName"
              class="max-w-full max-h-full object-contain"
            />
            <video
              v-else-if="item.media_type === 'video'"
              :src="item.url"
              controls
              class="max-w-full max-h-full"
              @click.stop
            />
            <audio
              v-else-if="item.media_type === 'audio'"
              :src="item.url"
              controls
              @click.stop
            />
          </div>
          <div class="px-1.5 py-1 border-t kt-v2-line/60 flex items-center justify-between gap-1 text-[10px]">
            <span class="flex items-center gap-1 truncate">
              <span class="w-2 h-2 rounded-full shrink-0" :style="{ background: item.color }" />
              <span class="truncate font-medium">{{ item.run_label }}</span>
            </span>
            <span class="num text-[var(--v2-ink-3)] shrink-0">st {{ item.step }}</span>
          </div>
        </div>
      </div>

      <div
        v-else
        class="h-full flex items-center justify-center text-xs text-[var(--v2-ink-3)]"
      >
        {{ loading ? "Loading media…" : "No media logged for selected runs." }}
      </div>
    </div>

    <!-- Step slider footer -->
    <div
      v-if="steps.length > 0"
      class="px-2.5 py-1.5 border-t kt-v2-line/70 flex items-center gap-2 text-[11px] shrink-0"
    >
      <span class="num text-[var(--v2-ink-2)] shrink-0">step {{ activeStep }}</span>
      <input
        v-if="steps.length > 1"
        v-model.number="stepIndex"
        type="range"
        min="0"
        :max="steps.length - 1"
        class="flex-1 min-w-0 accent-taaffeite h-1.5 cursor-pointer"
        title="Scrub step (hold Shift to sync all sliders)"
      />
      <span v-else class="flex-1" />
      <span class="num text-[10px] text-[var(--v2-ink-3)] shrink-0">
        {{ stepIndex + 1 }}/{{ steps.length }}
      </span>
      <button
        v-if="steps.length > 1"
        class="text-[10px] px-1.5 py-0.5 rounded border cursor-pointer shrink-0"
        :class="
          autoAdvance
            ? 'border-aquamarine text-aquamarine-shadow dark:text-aquamarine-light'
            : 'kt-v2-line text-[var(--v2-ink-3)]'
        "
        title="Auto-advance to newest step on live updates"
        @click="
          emit('patch-card', {
            autoAdvance: !autoAdvance,
            currentStep: !autoAdvance ? steps[steps.length - 1] : activeStep,
          })
        "
      >
        {{ autoAdvance ? "Live" : "Pinned" }}
      </button>
    </div>

    <!-- Lightbox overlay -->
    <Teleport to="body">
      <div
        v-if="lightbox"
        class="fixed inset-0 bg-black/75 z-50 flex items-center justify-center p-6"
        @click="lightbox = null"
      >
        <div class="kt-v2-float max-w-4xl max-h-[90vh] p-4 flex flex-col gap-2" @click.stop>
          <div class="flex items-center justify-between gap-4 text-xs">
            <span class="font-semibold">
              {{ lightbox.run_label }} · {{ card.mediaName }} (step {{ lightbox.step }})
            </span>
            <button class="btn-ghost" @click="lightbox = null">Close</button>
          </div>
          <img
            v-if="lightbox.media_type === 'image'"
            :src="lightbox.url"
            class="max-h-[75vh] object-contain rounded"
          />
          <div v-if="lightbox.caption" class="text-xs text-[var(--v2-ink-2)]">
            {{ lightbox.caption }}
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
