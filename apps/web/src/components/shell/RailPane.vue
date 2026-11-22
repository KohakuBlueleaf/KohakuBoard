<script setup>
import { ref, watch } from "vue"
import { useProjectsStore } from "@/stores/projects"
import { useTabsStore } from "@/stores/tabs"
import { useThemeStore } from "@/stores/theme"
import { statusColor } from "@/utils/palette"

const emit = defineEmits(["open-palette"])

const projects = useProjectsStore()
const tabs = useTabsStore()
const theme = useThemeStore()

const collapsed = ref(false)
const width = ref(236)
const dragging = ref(false)
const expandedProjects = ref({})

async function toggleProjectExpand(name) {
  expandedProjects.value[name] = !expandedProjects.value[name]
  if (expandedProjects.value[name] && projects.getRuns(name).length === 0) {
    await projects.loadRuns(name)
  }
}

async function selectProject(name) {
  expandedProjects.value[name] = true
  tabs.openProjectTab(name)
  await projects.loadRuns(name)
}

async function selectRun(projectName, run) {
  tabs.openRunTab(projectName, run.run_id, run.name)
  await projects.loadRunDetail(projectName, run.run_id)
}

watch(
  () => tabs.activeTab?.project,
  (proj) => {
    if (proj) expandedProjects.value[proj] = true
  },
  { immediate: true },
)

function startResize(ev) {
  ev.preventDefault()
  const startX = ev.clientX
  const startW = width.value
  dragging.value = true
  const onMove = (e) => {
    width.value = Math.max(176, Math.min(360, Math.round(startW + (e.clientX - startX))))
  }
  const onUp = () => {
    dragging.value = false
    window.removeEventListener("mousemove", onMove)
    window.removeEventListener("mouseup", onUp)
  }
  window.addEventListener("mousemove", onMove)
  window.addEventListener("mouseup", onUp)
}
</script>

<template>
  <div
    class="relative h-full shrink-0 select-none"
    :style="{ width: `${collapsed ? 48 : width}px` }"
    data-testid="rail-pane"
  >
    <!-- Collapsed icon strip (KT v2 RailPane style) -->
    <nav
      v-if="collapsed"
      class="h-full kt-v2-rail border-r kt-v2-edge flex flex-col items-center py-2.5 gap-1.5 overflow-hidden"
    >
      <button
        class="w-8 h-8 rounded-lg bg-iolite text-white flex items-center justify-center font-bold text-xs cursor-pointer"
        title="Expand sidebar"
        @click="collapsed = false"
      >
        K
      </button>
      <button
        class="w-8 h-8 rounded-md flex items-center justify-center text-[var(--v2-ink-2)] hover:bg-[var(--v2-tab-hover)] cursor-pointer"
        title="Expand sidebar"
        @click="collapsed = false"
      >
        <span class="i-carbon-side-panel-open text-base" />
      </button>
      <button
        class="w-8 h-8 rounded-md flex items-center justify-center text-[var(--v2-ink-2)] hover:bg-[var(--v2-tab-hover)] cursor-pointer"
        title="Command Palette (Ctrl+K)"
        @click="emit('open-palette')"
      >
        <span class="i-carbon-search text-base" />
      </button>

      <div class="w-6 border-t kt-v2-edge my-0.5" />

      <div class="flex-1 overflow-y-auto flex flex-col items-center gap-1">
        <button
          v-for="proj in projects.projects"
          :key="proj.name"
          class="w-8 h-8 rounded-md text-[11px] font-semibold flex items-center justify-center cursor-pointer"
          :class="
            tabs.activeTab?.project === proj.name
              ? 'bg-[var(--v2-card)] text-iolite dark:text-iolite-light shadow-sm'
              : 'text-[var(--v2-ink-2)] hover:bg-[var(--v2-tab-hover)]'
          "
          :title="`${proj.name} (${proj.run_count} runs)`"
          @click="selectProject(proj.name)"
        >
          {{ proj.name.slice(0, 2).toUpperCase() }}
        </button>
      </div>

      <button
        class="w-8 h-8 rounded-md flex items-center justify-center text-[var(--v2-ink-2)] hover:bg-[var(--v2-tab-hover)] cursor-pointer"
        data-testid="theme-toggle"
        :title="theme.isDark ? 'Switch to Warm Stone (Light)' : 'Switch to Warm Obsidian (Dark)'"
        @click="theme.toggle()"
      >
        <span :class="theme.isDark ? 'i-carbon-sun' : 'i-carbon-moon'" class="text-base" />
      </button>
    </nav>

    <!-- Expanded KT v2 RailPane with Projects -> Runs tree -->
    <nav
      v-else
      class="h-full kt-v2-rail border-r kt-v2-edge flex flex-col overflow-hidden"
    >
      <!-- Brand + Search + Collapse -->
      <div class="flex items-center gap-2 px-3 py-2.5 shrink-0">
        <span
          class="w-6 h-6 rounded-lg bg-iolite text-white flex items-center justify-center font-bold text-xs shrink-0"
        >
          K
        </span>
        <span class="text-sm flex-1 truncate">
          <span class="font-bold text-amber">Kohaku</span>
          <span class="font-light text-iolite dark:text-iolite-light">Board</span>
        </span>
        <button
          class="w-6 h-6 rounded flex items-center justify-center text-[var(--v2-ink-2)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-tab-hover)] cursor-pointer"
          title="Command Palette (Ctrl+K)"
          @click="emit('open-palette')"
        >
          <span class="i-carbon-search text-sm" />
        </button>
        <button
          class="w-6 h-6 rounded flex items-center justify-center text-[var(--v2-ink-2)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-tab-hover)] cursor-pointer"
          title="Collapse sidebar"
          @click="collapsed = true"
        >
          <span class="i-carbon-side-panel-close text-sm" />
        </button>
      </div>

      <div class="mx-2.5 border-t kt-v2-edge/70" />

      <!-- Projects & Runs tree -->
      <div class="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        <div
          class="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--v2-ink-3)] flex items-center justify-between"
        >
          <span>Projects & Runs</span>
          <button
            class="hover:text-[var(--v2-ink)] cursor-pointer"
            title="Refresh catalog"
            @click="projects.loadProjects()"
          >
            <span class="i-carbon-renew text-xs" />
          </button>
        </div>

        <div v-for="proj in projects.projects" :key="proj.name" class="space-y-0.5">
          <!-- Project Row -->
          <div
            class="w-full px-1.5 py-1.5 rounded-lg flex items-center justify-between gap-1 text-xs transition-colors cursor-pointer"
            :class="
              tabs.activeTab?.kind === 'project' && tabs.activeTab?.project === proj.name
                ? 'bg-[var(--v2-card)] text-iolite dark:text-iolite-light font-semibold shadow-sm'
                : 'text-[var(--v2-ink-2)] hover:bg-[var(--v2-tab-hover)] hover:text-[var(--v2-ink)]'
            "
            @click="selectProject(proj.name)"
          >
            <span class="flex items-center gap-1 min-w-0">
              <button
                class="w-4 h-4 rounded flex items-center justify-center hover:bg-[var(--v2-line)] shrink-0 cursor-pointer"
                :title="expandedProjects[proj.name] ? 'Collapse runs' : 'Expand runs'"
                @click.stop="toggleProjectExpand(proj.name)"
              >
                <span
                  :class="expandedProjects[proj.name] ? 'i-carbon-chevron-down' : 'i-carbon-chevron-right'"
                  class="text-[11px]"
                />
              </button>
              <span class="i-carbon-folder text-xs shrink-0 opacity-75" />
              <span class="truncate">{{ proj.name }}</span>
            </span>
            <span class="flex items-center gap-1 shrink-0">
              <span
                v-if="proj.running_count > 0"
                class="px-1.5 py-0.5 rounded-full text-[10px] num bg-aquamarine-light text-aquamarine-shadow dark:bg-aquamarine-shadow/50 dark:text-aquamarine-light"
              >
                {{ proj.running_count }} live
              </span>
              <span class="num text-[10px] text-[var(--v2-ink-3)]">{{ proj.run_count }}</span>
            </span>
          </div>

          <!-- Nested Run Items (direct entry to Run View!) -->
          <div
            v-if="expandedProjects[proj.name] && projects.getRuns(proj.name).length > 0"
            class="pl-4 space-y-0.5"
          >
            <button
              v-for="run in projects.getRuns(proj.name).slice(0, 20)"
              :key="run.run_id"
              class="w-full px-2 py-1 rounded-md flex items-center justify-between gap-1.5 text-[11px] transition-colors cursor-pointer"
              :class="
                tabs.activeTab?.kind === 'run' &&
                tabs.activeTab?.project === proj.name &&
                tabs.activeTab?.runId === run.run_id
                  ? 'bg-[var(--v2-card)] text-iolite dark:text-iolite-light font-semibold shadow-sm'
                  : 'text-[var(--v2-ink-2)] hover:bg-[var(--v2-tab-hover)] hover:text-[var(--v2-ink)]'
              "
              :data-testid="`rail-run-${run.run_id}`"
              @click="selectRun(proj.name, run)"
            >
              <span class="flex items-center gap-1.5 min-w-0">
                <span
                  class="w-1.5 h-1.5 rounded-full shrink-0"
                  :style="{ background: statusColor(run.state) }"
                />
                <span class="truncate">{{ run.name }}</span>
              </span>
              <span class="num text-[10px] text-[var(--v2-ink-3)] shrink-0">{{ run.run_id.slice(0, 6) }}</span>
            </button>
          </div>
        </div>

        <div
          v-if="projects.projects.length === 0 && !projects.loadingProjects"
          class="px-2.5 py-4 text-xs text-[var(--v2-ink-3)]"
        >
          No projects yet. Log a run with <code>kobo.init()</code>.
        </div>
      </div>

      <!-- Footer: Host/SSE chip + Theme toggle -->
      <div class="mx-2.5 border-t kt-v2-edge/70" />
      <div class="px-3 py-2 flex items-center justify-between gap-2 shrink-0">
        <span class="flex items-center gap-1.5 text-[11px] text-[var(--v2-ink-2)]">
          <span
            class="w-2 h-2 rounded-full"
            :class="projects.sseConnected ? 'bg-aquamarine' : 'bg-warm-400'"
          />
          <span>{{ projects.sseConnected ? "Live SSE" : "Local" }}</span>
        </span>

        <button
          class="w-7 h-7 rounded-md flex items-center justify-center text-[var(--v2-ink-2)] hover:text-[var(--v2-ink)] hover:bg-[var(--v2-tab-hover)] cursor-pointer"
          data-testid="theme-toggle"
          :title="theme.isDark ? 'Switch to Warm Stone (Light)' : 'Switch to Warm Obsidian (Dark)'"
          @click="theme.toggle()"
        >
          <span :class="theme.isDark ? 'i-carbon-sun' : 'i-carbon-moon'" class="text-sm" />
        </button>
      </div>
    </nav>

    <!-- 4px draggable right edge resize handle -->
    <div
      v-if="!collapsed"
      class="absolute top-0 right-0 h-full w-[4px] cursor-col-resize hover:bg-iolite/35"
      :class="dragging ? 'bg-iolite/55' : ''"
      title="Drag to resize · double-click to reset"
      @mousedown="startResize"
      @dblclick="width = 236"
    />
  </div>
</template>
