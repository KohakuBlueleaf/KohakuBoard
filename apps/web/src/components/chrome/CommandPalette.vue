<script setup>
import { computed, nextTick, ref, watch } from "vue"
import { useProjectsStore } from "@/stores/projects"
import { useTabsStore } from "@/stores/tabs"
import { useThemeStore } from "@/stores/theme"

const props = defineProps({
  open: { type: Boolean, default: false },
})
const emit = defineEmits(["close"])

const projects = useProjectsStore()
const tabs = useTabsStore()
const theme = useThemeStore()

const query = ref("")
const inputEl = ref(null)

watch(
  () => props.open,
  async (val) => {
    if (val) {
      query.value = ""
      await nextTick()
      inputEl.value?.focus()
    }
  },
)

const items = computed(() => {
  const list = []
  for (const p of projects.projects) {
    list.push({
      id: `proj:${p.name}`,
      group: "Project",
      label: p.name,
      sub: `${p.run_count} runs`,
      run: async () => {
        tabs.openProjectTab(p.name)
        await projects.loadRuns(p.name)
      },
    })
    for (const r of projects.getRuns(p.name).slice(0, 20)) {
      list.push({
        id: `run:${p.name}/${r.run_id}`,
        group: "Run",
        label: `${r.name} (${r.run_id})`,
        sub: `${p.name} · ${r.state}`,
        run: () => tabs.openRunTab(p.name, r.run_id, r.name),
      })
    }
  }
  if (tabs.activeTab) {
    for (const [id, label] of [
      ["charts", "Charts"],
      ["table", "Runs Table"],
      ["media", "Media"],
      ["distributions", "Distributions"],
      ["tables", "Tables"],
      ["logs", "Console Logs"],
    ]) {
      if (tabs.activeTab.kind === "run" && id === "table") continue
      list.push({
        id: `view:${id}`,
        group: "View",
        label: `Switch view: ${label}`,
        sub: tabs.activeTab.title,
        run: () => tabs.setTabView(tabs.activeTab.id, id),
      })
    }
  }
  list.push({
    id: "action:theme",
    group: "Theme",
    label: theme.isDark ? "Switch to Warm Stone (Light)" : "Switch to Warm Obsidian (Dark)",
    sub: "Appearance",
    run: () => theme.toggle(),
  })

  const q = query.value.trim().toLowerCase()
  if (!q) return list.slice(0, 14)
  return list.filter(
    (it) => it.label.toLowerCase().includes(q) || it.sub.toLowerCase().includes(q),
  )
})

async function execute(item) {
  emit("close")
  await item.run()
}
</script>

<template>
  <div
    v-if="open"
    class="fixed inset-0 bg-black/45 z-50 flex items-start justify-center pt-20 px-4"
    @click="emit('close')"
  >
    <div
      class="kt-v2-float w-full max-w-lg overflow-hidden"
      data-testid="command-palette"
      @click.stop
    >
      <div class="p-3 border-b kt-v2-line flex items-center gap-2">
        <span class="i-carbon-search text-sm text-[var(--v2-ink-3)]" />
        <input
          ref="inputEl"
          v-model="query"
          type="text"
          placeholder="Search projects, runs, views, theme..."
          class="w-full text-sm bg-transparent text-[var(--v2-ink)] outline-none"
          @keydown.esc="emit('close')"
          @keydown.enter="items[0] && execute(items[0])"
        />
      </div>
      <div class="max-h-72 overflow-y-auto py-1">
        <button
          v-for="item in items"
          :key="item.id"
          class="kt-row w-full px-3 py-2 flex items-center justify-between text-left text-xs cursor-pointer"
          @click="execute(item)"
        >
          <span class="flex items-center gap-2 min-w-0">
            <span class="px-1.5 py-0.5 rounded text-[10px] bg-[var(--v2-panel)] text-[var(--v2-ink-2)]">
              {{ item.group }}
            </span>
            <span class="font-medium truncate">{{ item.label }}</span>
          </span>
          <span class="num text-[10px] text-[var(--v2-ink-3)] shrink-0">{{ item.sub }}</span>
        </button>
      </div>
    </div>
  </div>
</template>
