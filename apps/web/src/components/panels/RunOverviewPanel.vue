<script setup>
import { computed, ref, watch } from "vue"
import { useProjectsStore } from "@/stores/projects"
import { useWorkspaceStore } from "@/stores/workspace"
import { api } from "@/utils/api"
import { statusColor } from "@/utils/palette"
import { formatBytes, formatDuration, formatNumber } from "@/utils/format"

const props = defineProps({
  scopeId: { type: String, required: true },
  project: { type: String, required: true },
  lockedRunId: { type: String, default: null },
})

const projects = useProjectsStore()
const workspace = useWorkspaceStore()

const artifacts = ref([])
const alerts = ref([])
const notesDraft = ref("")
const groupDraft = ref("")
const tagsDraft = ref("")
const savingMeta = ref(false)

const allRuns = computed(() => projects.getRuns(props.project))
const targetRun = computed(() => {
  if (props.lockedRunId) return allRuns.value.find((r) => r.run_id === props.lockedRunId) || null
  const vis = workspace.visibleRuns(props.scopeId, allRuns.value)
  return vis[0] || allRuns.value[0] || null
})

const detail = computed(() =>
  targetRun.value ? projects.getDetail(props.project, targetRun.value.run_id) : null,
)

async function loadExtras() {
  if (!targetRun.value) return
  const d = await projects.loadRunDetail(props.project, targetRun.value.run_id)
  notesDraft.value = d?.info?.notes || ""
  groupDraft.value = d?.info?.group || ""
  tagsDraft.value = (d?.info?.tags || []).join(", ")
  const [arts, alts] = await Promise.all([
    api.artifacts(props.project, targetRun.value.run_id),
    api.alerts(props.project, targetRun.value.run_id),
  ])
  artifacts.value = arts
  alerts.value = alts
}

async function saveMetadata() {
  if (!targetRun.value) return
  savingMeta.value = true
  try {
    const tags = tagsDraft.value
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean)
    await projects.patchRun(props.project, targetRun.value.run_id, {
      notes: notesDraft.value,
      group: groupDraft.value.trim() || null,
      tags,
    })
  } finally {
    savingMeta.value = false
  }
}

watch(targetRun, loadExtras, { immediate: true })
</script>

<template>
  <div class="h-full flex flex-col kt-v2-canvas overflow-y-auto p-3 space-y-3" data-testid="run-overview">
    <template v-if="detail">
      <div class="kt-v2-card p-3 space-y-2">
        <div class="flex items-center justify-between gap-2">
          <div class="min-w-0">
            <div class="text-sm font-semibold truncate">{{ detail.info.name }}</div>
            <div class="text-[11px] num text-[var(--v2-ink-3)]">
              {{ detail.info.project }} / {{ detail.info.run_id }} ·
              {{ formatDuration(detail.info.created_at_ms, detail.info.finished_at_ms) }}
            </div>
          </div>
          <span
            class="px-2 py-0.5 rounded-full text-[11px] font-medium text-white shrink-0"
            :style="{ background: statusColor(detail.info.state) }"
          >
            {{ detail.info.state }}
          </span>
        </div>

        <div class="grid grid-cols-2 gap-2 text-xs pt-1 border-t kt-v2-line">
          <div>
            <span class="text-[var(--v2-ink-3)]">run.db:</span>
            <span class="num ml-1">{{ formatBytes(detail.storage?.run_db_bytes) }}</span>
          </div>
          <div>
            <span class="text-[var(--v2-ink-3)]">blobs.db:</span>
            <span class="num ml-1">{{ formatBytes(detail.storage?.blobs_db_bytes) }}</span>
          </div>
          <div v-if="detail.env?.hardware?.cpu_count">
            <span class="text-[var(--v2-ink-3)]">CPU:</span>
            <span class="num ml-1">{{ detail.env.hardware.cpu_count }} cores</span>
          </div>
          <div v-if="detail.env?.python">
            <span class="text-[var(--v2-ink-3)]">Python:</span>
            <span class="num ml-1">{{ detail.env.python }}</span>
          </div>
        </div>
      </div>

      <!-- Editable Group / Tags / Notes -->
      <div class="kt-v2-card p-3 space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-xs font-semibold">Metadata & Notes</span>
          <button class="btn-primary text-[11px] py-1 px-2" :disabled="savingMeta" @click="saveMetadata">
            {{ savingMeta ? "Saving…" : "Save" }}
          </button>
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs">
          <input
            v-model="groupDraft"
            type="text"
            placeholder="Group"
            class="px-2 py-1 rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] outline-none focus:border-iolite"
          />
          <input
            v-model="tagsDraft"
            type="text"
            placeholder="Tags (comma-sep)"
            class="px-2 py-1 rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] outline-none focus:border-iolite"
          />
        </div>
        <textarea
          v-model="notesDraft"
          rows="3"
          placeholder="Add experiment notes..."
          class="w-full p-2 text-xs rounded-lg border kt-v2-line bg-[var(--v2-canvas)] text-[var(--v2-ink)] outline-none focus:border-iolite"
        />
      </div>

      <!-- Config -->
      <div class="kt-v2-card p-3">
        <div class="text-xs font-semibold mb-2 text-sapphire dark:text-sapphire-light">Config</div>
        <div class="space-y-1 text-xs">
          <div
            v-for="(val, k) in detail.config"
            :key="k"
            class="flex items-center justify-between gap-2 py-0.5 border-b kt-v2-line/50"
          >
            <span class="text-[var(--v2-ink-2)]">{{ k }}</span>
            <span class="num truncate max-w-[65%]">{{
              typeof val === "object" ? JSON.stringify(val) : val
            }}</span>
          </div>
          <div v-if="Object.keys(detail.config || {}).length === 0" class="text-[var(--v2-ink-3)]">
            Empty config
          </div>
        </div>
      </div>

      <!-- Summary -->
      <div v-if="Object.keys(detail.summary || {}).length > 0" class="kt-v2-card p-3">
        <div class="text-xs font-semibold mb-2 text-aquamarine-shadow dark:text-aquamarine-light">
          Summary
        </div>
        <div class="space-y-1 text-xs">
          <div
            v-for="(s, k) in detail.summary"
            :key="k"
            class="flex items-center justify-between py-0.5 border-b kt-v2-line/50"
          >
            <span class="truncate">{{ k }}</span>
            <span class="num">{{ formatNumber(s.last, 5) }}</span>
          </div>
        </div>
      </div>

      <!-- Alerts & Artifacts -->
      <div v-if="alerts.length > 0" class="kt-v2-card p-3 space-y-1.5">
        <div class="text-xs font-semibold text-amber-shadow dark:text-amber-light">Alerts</div>
        <div v-for="al in alerts" :key="al.id" class="text-xs border-b kt-v2-line/50 pb-1">
          <div class="font-medium">[{{ al.level }}] {{ al.title }} (st {{ al.step }})</div>
          <div class="text-[var(--v2-ink-2)]">{{ al.text }}</div>
        </div>
      </div>

      <div v-if="artifacts.length > 0" class="kt-v2-card p-3 space-y-1.5">
        <div class="text-xs font-semibold text-taaffeite-shadow dark:text-taaffeite-light">
          Artifacts
        </div>
        <div v-for="ar in artifacts" :key="ar.id" class="text-xs border-b kt-v2-line/50 pb-1">
          <div class="font-medium">{{ ar.name }} ({{ ar.kind }})</div>
          <div class="num text-[10px] text-[var(--v2-ink-3)] truncate">{{ ar.uri }}</div>
        </div>
      </div>
    </template>

    <div v-else class="h-32 flex items-center justify-center text-xs text-[var(--v2-ink-3)]">
      Select a run to inspect.
    </div>
  </div>
</template>
