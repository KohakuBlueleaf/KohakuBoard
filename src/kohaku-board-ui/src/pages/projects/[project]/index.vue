<script setup>
import { ElMessage } from "element-plus";
import { Loading } from "@element-plus/icons-vue";
import { VueDraggable } from "vue-draggable-plus";
import ConfigurableChartCard from "@/components/ConfigurableChartCard.vue";
import RunSelectionList from "@/components/RunSelectionList.vue";
import { useAnimationPreference } from "@/composables/useAnimationPreference";
import { useHoverSync } from "@/composables/useHoverSync";

const route = useRoute();
const { animationsEnabled } = useAnimationPreference();
const { hoverSyncEnabled, toggleHoverSync } = useHoverSync();

const projectName = computed(() => route.params.project);

const legendModeKey = computed(
  () => `legend-mode-${projectName.value ?? "default"}`,
);
const legendMode = ref("annotation");

watch(
  () => projectName.value,
  () => {
    legendMode.value =
      localStorage.getItem(legendModeKey.value) || "annotation";
  },
  { immediate: true },
);

watch(legendMode, (value) => {
  localStorage.setItem(legendModeKey.value, value);
});

// Run selection state
const allRuns = ref([]);
const selectedRunIds = ref(new Set());
const currentPage = ref(1);
const runsPerPage = 20;
const loading = ref(false);

// Base color palette for runs
const RUN_COLORS = [
  "#FF6B6B",
  "#4ECDC4",
  "#45B7D1",
  "#FFA07A",
  "#98D8C8",
  "#F7DC6F",
  "#BB8FCE",
  "#85C1E2",
  "#F8B88B",
  "#52BE80",
  "#E74C3C",
  "#3498DB",
  "#9B59B6",
  "#1ABC9C",
  "#F39C12",
  "#E67E22",
  "#95A5A6",
  "#34495E",
  "#16A085",
  "#27AE60",
];

/**
 * Get deterministic color for a run
 * Strategy: Use ONLY run_id hash for full determinism
 *
 * - Base color selected from palette using run_id hash (always stable)
 * - Slight HSL jitter based on run_id hash (makes each run unique)
 * - Same run_id ALWAYS gets same color, regardless of run order/deletion
 */
function getRunColor(runId) {
  // Generate hash from run_id for palette selection
  let hash = 0;
  for (let i = 0; i < runId.length; i++) {
    const char = runId.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }

  // Use hash to select base color from palette (deterministic)
  const paletteIndex = Math.abs(hash) % RUN_COLORS.length;
  const baseColor = RUN_COLORS[paletteIndex];

  // Use the same hash to generate jitter values (-15 to +15 for hue, -10% to +10% for saturation/lightness)
  const hueJitter = (hash % 30) - 15;
  const satJitter = (((hash >> 8) % 20) - 10) / 100;
  const lightJitter = (((hash >> 16) % 20) - 10) / 100;

  // Parse base color to RGB
  const hex = baseColor.replace("#", "");
  const r = parseInt(hex.substr(0, 2), 16) / 255;
  const g = parseInt(hex.substr(2, 2), 16) / 255;
  const b = parseInt(hex.substr(4, 2), 16) / 255;

  // Convert RGB to HSL
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h,
    s,
    l = (max + min) / 2;

  if (max === min) {
    h = s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

    switch (max) {
      case r:
        h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
        break;
      case g:
        h = ((b - r) / d + 2) / 6;
        break;
      case b:
        h = ((r - g) / d + 4) / 6;
        break;
    }
  }

  // Apply jitter
  h = (h * 360 + hueJitter) % 360;
  s = Math.max(0, Math.min(1, s + satJitter));
  l = Math.max(0, Math.min(1, l + lightJitter));

  // Convert HSL back to RGB
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };

  let r2, g2, b2;
  if (s === 0) {
    r2 = g2 = b2 = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r2 = hue2rgb(p, q, h / 360 + 1 / 3);
    g2 = hue2rgb(p, q, h / 360);
    b2 = hue2rgb(p, q, h / 360 - 1 / 3);
  }

  // Convert back to hex
  const toHex = (x) => {
    const hex = Math.round(x * 255).toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  };

  return `#${toHex(r2)}${toHex(g2)}${toHex(b2)}`;
}

// Multi-run data cache (run_id -> metric -> sparse array)
const multiRunDataCache = ref({});
const availableMetrics = ref([]);
const runSummaries = ref({});

// UI state (same as [id].vue)
const tabs = ref([{ name: "Metrics", cards: [] }]);
const activeTab = ref("Metrics");
const nextCardId = ref(1);
const isEditingTabs = ref(false);
const isUpdating = ref(false);
const showAddTabDialog = ref(false);
const newTabName = ref("");
const showGlobalSettings = ref(false);
const showAddChartDialog = ref(false);
const newChartType = ref("line");
const newChartValue = ref([]);
const removedMetrics = ref(new Set());
const isInitializing = ref(true);
const chartsPerPage = ref(12);

// Pagination for WebGL context limit
const currentChartPage = ref(0);
const isMobile = ref(window.innerWidth <= 900);

// Responsive detection
const mediaQuery = window.matchMedia("(max-width: 900px)");
const handleMediaChange = (e) => {
  isMobile.value = e.matches;
  if (!isMobile.value) {
    // Expanded to desktop - always show sidebar
    isSidebarCollapsed.value = false;
  } else {
    // Shrunk to mobile - collapse sidebar
    isSidebarCollapsed.value = true;
  }
};

onMounted(() => {
  mediaQuery.addEventListener("change", handleMediaChange);
});

onUnmounted(() => {
  mediaQuery.removeEventListener("change", handleMediaChange);
});

// Sidebar state
const sidebarWidth = ref(300); // Default 300px
const minSidebarWidth = 200;
const maxSidebarWidth = 600;
const isSidebarCollapsed = ref(true); // Start collapsed on mobile
const isResizingSidebar = ref(false);

// Global settings
const globalSettings = ref({
  xAxis: "global_step",
  smoothing: "disabled",
  smoothingValue: 0.9,
  downsampleRate: -1,
});

const PROJECT_LAYOUT_KEY_PREFIX = "project-dashboard-layout";
const LEGACY_PROJECT_LAYOUT_PREFIX = "project-layout";

const storageKey = computed(
  () => `${PROJECT_LAYOUT_KEY_PREFIX}-${route.params.project}`,
);
const legacyStorageKey = computed(
  () => `${LEGACY_PROJECT_LAYOUT_PREFIX}-${route.params.project}`,
);

function parseProjectLayout(rawValue, label) {
  if (!rawValue) return null;
  try {
    const parsed = JSON.parse(rawValue);
    if (!isProjectLayoutPayload(parsed)) {
      return null;
    }
    if (!Array.isArray(parsed.hiddenRunIds)) {
      parsed.hiddenRunIds = [];
    }
    if (!Array.isArray(parsed.removedMetrics)) {
      parsed.removedMetrics = [];
    }
    return parsed;
  } catch (error) {
    console.warn(`[ProjectLayout] Failed to parse ${label}:`, error);
    return null;
  }
}

function isProjectLayoutPayload(payload) {
  if (!payload || typeof payload !== "object") return false;
  if (!Array.isArray(payload.tabs)) return false;
  if (!("hiddenRunIds" in payload)) return false;
  return true;
}

function loadProjectLayoutFromStorage() {
  const current = parseProjectLayout(
    localStorage.getItem(storageKey.value),
    storageKey.value,
  );
  if (current) {
    return current;
  }

  if (legacyStorageKey.value === storageKey.value) {
    return null;
  }

  const legacy = parseProjectLayout(
    localStorage.getItem(legacyStorageKey.value),
    legacyStorageKey.value,
  );
  if (legacy) {
    localStorage.setItem(storageKey.value, JSON.stringify(legacy));
    localStorage.removeItem(legacyStorageKey.value);
    return legacy;
  }
  return null;
}

const AXIS_ONLY_METRICS = new Set([
  "step",
  "global_step",
  "timestamp",
  "walltime",
  "relative_walltime",
]);

// Custom run colors (saved by user, overrides hash-based defaults)
// Now saved with layout instead of separate storage
const customRunColors = ref({});

// Run colors map - fully deterministic hash-based colors
// Map by both run_id AND run name for easy lookup
const runColors = computed(() => {
  const colors = {};
  allRuns.value.forEach((run) => {
    colors[run.run_id] =
      customRunColors.value[run.run_id] || getRunColor(run.run_id);
  });
  return colors;
});

// Run names map - map run_id to combined label for display
const runNames = computed(() => {
  const names = {};
  allRuns.value.forEach((run) => {
    const useName = legendMode.value === "name" && run.name;
    names[run.run_id] = useName ? run.name : run.run_id;
  });
  return names;
});

// Displayed runs (first 10 selected, which are the latest since allRuns is sorted)
const displayedRuns = computed(() => {
  const selected = Array.from(selectedRunIds.value);

  // Map to actual run objects to preserve sort order
  const selectedRuns = selected
    .map((runId) => allRuns.value.find((r) => r.run_id === runId))
    .filter(Boolean);

  // Take first 10 (which are latest since allRuns is sorted by date desc)
  return selectedRuns.slice(0, 10);
});

// Paginated runs for sidebar
const paginatedRuns = computed(() => {
  const start = (currentPage.value - 1) * runsPerPage;
  return allRuns.value.slice(start, start + runsPerPage);
});

const totalRuns = computed(() => allRuns.value.length);
const totalPages = computed(() => Math.ceil(totalRuns.value / runsPerPage));

let pollInterval = null;

// Fetch all runs for this project
async function fetchRuns() {
  loading.value = true;
  try {
    const response = await fetch(`/api/projects/${projectName.value}/runs`);

    if (!response.ok) {
      throw new Error(`Failed to fetch runs: ${response.status}`);
    }

    const data = await response.json();
    // Sort runs by created_at (latest first)
    allRuns.value = (data.runs || []).sort((a, b) => {
      const dateA = new Date(a.created_at || 0);
      const dateB = new Date(b.created_at || 0);
      return dateB - dateA; // Latest first
    });

    // Load saved hidden runs, default to all visible
    const savedLayout = loadProjectLayoutFromStorage();
    const hiddenRunIds = savedLayout?.hiddenRunIds || [];

    // Select all runs except hidden ones
    const allRunIds = allRuns.value.map((r) => r.run_id);
    selectedRunIds.value = new Set(
      allRunIds.filter((runId) => !hiddenRunIds.includes(runId)),
    );

    // Initialize project view
    await initializeProject();
  } catch (error) {
    console.error("Failed to fetch runs:", error);
    ElMessage.error("Failed to fetch project runs");
  } finally {
    loading.value = false;
  }
}

// Poll for new runs and run updates in background
async function pollRuns() {
  try {
    console.log("[Polling] Starting poll check...");
    const response = await fetch(`/api/projects/${projectName.value}/runs`);
    if (!response.ok) {
      console.log(`[Polling] Failed to fetch runs: ${response.status}`);
      return;
    }

    const data = await response.json();
    const newRuns = (data.runs || []).sort((a, b) => {
      const dateA = new Date(a.created_at || 0);
      const dateB = new Date(b.created_at || 0);
      return dateB - dateA;
    });

    console.log(`[Polling] Fetched ${newRuns.length} runs from server`);
    console.log(`[Polling] Current allRuns count:`, allRuns.value.length);
    console.log(
      `[Polling] Current visible runs:`,
      Array.from(selectedRunIds.value),
    );

    // Check if there are new runs OR if any VISIBLE run has been updated
    const oldRunIds = new Set(allRuns.value.map((r) => r.run_id));
    const oldRunMap = new Map(
      allRuns.value.map((r) => [r.run_id, r.updated_at]),
    );

    console.log(`[Polling] Old runs map:`, Object.fromEntries(oldRunMap));
    console.log(
      `[Polling] New runs updated_at:`,
      newRuns.map((r) => ({ id: r.run_id, updated_at: r.updated_at })),
    );

    const hasNewRuns = newRuns.some((r) => !oldRunIds.has(r.run_id));
    console.log(`[Polling] Has new runs: ${hasNewRuns}`);

    // Check for deleted runs (runs that existed before but not in new list)
    const newRunIds = new Set(newRuns.map((r) => r.run_id));
    const deletedRuns = allRuns.value.filter((r) => !newRunIds.has(r.run_id));
    const hasDeletedRuns = deletedRuns.length > 0;
    console.log(`[Polling] Has deleted runs: ${hasDeletedRuns}`);
    if (hasDeletedRuns) {
      console.log(
        `[Polling] Deleted runs:`,
        deletedRuns.map((r) => r.run_id),
      );
    }

    // Only check updates for visible (selected) runs
    const updatedRunsList = [];
    for (const r of newRuns) {
      // Skip if run is not visible
      if (!selectedRunIds.value.has(r.run_id)) {
        console.log(`[Polling] Skipping hidden run: ${r.run_id}`);
        continue;
      }

      const oldUpdatedAt = oldRunMap.get(r.run_id);
      console.log(`[Polling] Checking run ${r.run_id}:`);
      console.log(
        `[Polling]   - Old updated_at: ${oldUpdatedAt} (type: ${typeof oldUpdatedAt})`,
      );
      console.log(
        `[Polling]   - New updated_at: ${r.updated_at} (type: ${typeof r.updated_at})`,
      );
      console.log(`[Polling]   - Are equal: ${oldUpdatedAt === r.updated_at}`);

      // Check if updated_at changed (including first time when old is null)
      // If old is null/undefined, it's the first poll after page load - treat as update if new exists
      // If both exist and are different, it's an update
      if (r.updated_at && oldUpdatedAt !== r.updated_at) {
        console.log(`[Polling]   - UPDATE DETECTED!`);
        updatedRunsList.push({
          run_id: r.run_id,
          old: oldUpdatedAt,
          new: r.updated_at,
        });
      } else {
        console.log(`[Polling]   - No change detected`);
      }
    }

    const hasUpdatedRuns = updatedRunsList.length > 0;
    console.log(`[Polling] Has updated visible runs: ${hasUpdatedRuns}`);
    if (hasUpdatedRuns) {
      console.log("[Polling] Updated runs:", updatedRunsList);
    }

    if (hasNewRuns || hasDeletedRuns || hasUpdatedRuns) {
      if (hasNewRuns || hasDeletedRuns) {
        console.log(
          "[Polling] Runs added/deleted, updating list and chart data",
        );
        allRuns.value = newRuns;

        // Load saved hidden runs
        const savedLayout = loadProjectLayoutFromStorage();
        const hiddenRunIds = savedLayout?.hiddenRunIds || [];

        // Select all runs except hidden ones (new runs default to visible)
        const allRunIds = allRuns.value.map((r) => r.run_id);
        selectedRunIds.value = new Set(
          allRunIds.filter((runId) => !hiddenRunIds.includes(runId)),
        );

        // Full re-initialization for new/deleted runs
        console.log("[Polling] Calling initializeProject()");
        await initializeProject();
        console.log("[Polling] initializeProject() completed");
      } else if (hasUpdatedRuns) {
        console.log("[Polling] Existing runs updated, refreshing chart data");
        allRuns.value = newRuns;
        // Just refresh metric data, don't rebuild UI
        console.log("[Polling] Calling refreshMetricData()");
        await refreshMetricData();
        console.log("[Polling] refreshMetricData() completed");
      }
    } else {
      console.log("[Polling] No changes detected, skipping refresh");
    }
  } catch (error) {
    console.error("[Polling] Error:", error);
  }
}

// Refresh metric data without rebuilding UI (for polling updates)
async function refreshMetricData() {
  try {
    console.log("[Refresh] Refreshing metric data for updated runs");

    if (displayedRuns.value.length === 0) {
      console.log("[Refresh] No displayed runs, skipping");
      return;
    }

    // Fetch batch summaries to check for new metrics
    const runIds = displayedRuns.value.map((r) => r.run_id);
    console.log("[Refresh] Fetching summaries for runs:", runIds);

    const response = await fetch(
      `/api/projects/${projectName.value}/runs/batch/summary`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: runIds }),
      },
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch summaries: ${response.status}`);
    }

    runSummaries.value = await response.json();
    console.log("[Refresh] Received summaries:", runSummaries.value);

    // Check if there are new metrics
    const allMetricsSet = new Set();
    for (const runId of runIds) {
      const summary = runSummaries.value[runId];
      if (summary?.available_data?.scalars) {
        for (const metric of summary.available_data.scalars) {
          allMetricsSet.add(metric);
        }
      }
    }

    const newMetrics = Array.from(allMetricsSet).sort();
    if (availableMetrics.value.includes("timestamp")) {
      newMetrics.push("walltime");
      newMetrics.push("relative_walltime");
    }

    console.log("[Refresh] Available metrics:", newMetrics);
    availableMetrics.value = newMetrics;

    // Refetch all metrics for current tab with force refresh
    console.log("[Refresh] Calling fetchMetricsForTab(true)");
    await fetchMetricsForTab(true);

    console.log("[Refresh] Metric data refreshed successfully");
  } catch (error) {
    console.error("[Refresh] Failed to refresh metric data:", error);
  }
}

async function toggleRunSelection(runId) {
  const wasVisible = selectedRunIds.value.has(runId);

  if (wasVisible) {
    selectedRunIds.value.delete(runId);
  } else {
    selectedRunIds.value.add(runId);
  }

  // Trigger reactivity
  selectedRunIds.value = new Set(selectedRunIds.value);

  // If making visible, trigger immediate poll + refresh
  if (!wasVisible) {
    console.log(`Run ${runId} made visible, triggering refresh`);
    // Poll to get latest updated_at
    await pollRuns();
    // If pollRuns didn't trigger refresh (no updates detected), manually refresh
    if (displayedRuns.value.some((r) => r.run_id === runId)) {
      await refreshMetricData();
    }
  }
}

function updateRunColor(runId, color) {
  customRunColors.value[runId] = color;
  // Force reactivity
  customRunColors.value = { ...customRunColors.value };
  // Save with layout (includes custom colors)
  saveLayout();
}

function selectAllOnPage() {
  for (const run of paginatedRuns.value) {
    selectedRunIds.value.add(run.run_id);
  }
  selectedRunIds.value = new Set(selectedRunIds.value);
}

function deselectAll() {
  selectedRunIds.value.clear();
  selectedRunIds.value = new Set(selectedRunIds.value);
}

function handlePageChange(page) {
  currentPage.value = page;
}

// Watch for changes to displayed runs
watch(
  displayedRuns,
  async () => {
    if (!isInitializing.value) {
      await initializeProject();
    }
  },
  { deep: true },
);

// Initialize project view (fetch summaries and build tabs)
async function initializeProject() {
  try {
    isInitializing.value = true;

    if (displayedRuns.value.length === 0) {
      tabs.value = [{ name: "Metrics", cards: [] }];
      multiRunDataCache.value = {};
      availableMetrics.value = [];
      isInitializing.value = false;
      return;
    }

    // Fetch batch summaries
    const runIds = displayedRuns.value.map((r) => r.run_id);
    const response = await fetch(
      `/api/projects/${projectName.value}/runs/batch/summary`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: runIds }),
      },
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch summaries: ${response.status}`);
    }

    runSummaries.value = await response.json();

    // Aggregate all metrics across runs
    const allMetricsSet = new Set();
    for (const runId of runIds) {
      const summary = runSummaries.value[runId];
      if (summary?.available_data?.scalars) {
        for (const metric of summary.available_data.scalars) {
          allMetricsSet.add(metric);
        }
      }
    }

    availableMetrics.value = Array.from(allMetricsSet).sort();

    // Add computed metrics
    if (availableMetrics.value.includes("timestamp")) {
      availableMetrics.value.push("walltime");
      availableMetrics.value.push("relative_walltime");
    }

    // Try to load saved layout
    const savedLayout = loadProjectLayoutFromStorage();

    if (savedLayout) {
      activeTab.value = savedLayout.activeTab || "Metrics";
      nextCardId.value = savedLayout.nextCardId || 1;

      if (savedLayout.globalSettings) {
        globalSettings.value = savedLayout.globalSettings;
      }

      if (savedLayout.customRunColors) {
        customRunColors.value = savedLayout.customRunColors;
      }

      if (savedLayout.sidebarWidth) {
        sidebarWidth.value = savedLayout.sidebarWidth;
      }

      removedMetrics.value = new Set(savedLayout.removedMetrics || []);
    } else {
      removedMetrics.value = new Set();
    }

    // Build default cards if no saved layout
    if (savedLayout?.tabs) {
      tabs.value = sanitizeProjectTabs(savedLayout.tabs);
    } else {
      buildDefaultTabs();
    }

    const addedDefaults = ensureDefaultCardsPresent();
    if (addedDefaults) {
      saveLayout();
    }

    // Fetch all metrics FIRST
    await fetchMetricsForTab();

    // THEN update cards to match current displayed runs (after data is loaded)
    updateCardsForCurrentRuns();

    isInitializing.value = false;
  } catch (error) {
    console.error("Failed to initialize project:", error);
    isInitializing.value = false;
  }
}

// Update existing cards - keep yMetrics as base metrics only
function updateCardsForCurrentRuns() {
  for (const tab of tabs.value) {
    for (const card of tab.cards) {
      if (
        card.config.type === "line" &&
        card.config.yMetrics &&
        card.config.yMetrics.length > 0
      ) {
        // Strip run suffixes if they exist (for backwards compatibility with old layouts)
        const baseMetrics = card.config.yMetrics.map((yMetric) => {
          return yMetric.includes(" (")
            ? yMetric.substring(0, yMetric.lastIndexOf(" ("))
            : yMetric;
        });

        // Remove duplicates
        card.config.yMetrics = [...new Set(baseMetrics)];

        if (Array.isArray(card.defaultMetrics)) {
          card.defaultMetrics = card.defaultMetrics
            .map(stripMetricSuffix)
            .filter((metric) => card.config.yMetrics.includes(metric));
        } else if (card.config.yMetrics.length === 1) {
          card.defaultMetrics = [card.config.yMetrics[0]];
        } else {
          card.defaultMetrics = [];
        }

        console.log(
          `Updated card ${card.id} yMetrics to base metrics only:`,
          card.config.yMetrics,
        );
      }
    }
  }
}

function stripMetricSuffix(metric) {
  if (typeof metric !== "string") return metric;
  const idx = metric.lastIndexOf(" (");
  return idx === -1 ? metric : metric.substring(0, idx);
}

function sanitizeProjectTabs(rawTabs) {
  if (!Array.isArray(rawTabs)) return [{ name: "Metrics", cards: [] }];
  const sanitized = [];

  rawTabs.forEach((tab, tabIndex) => {
    const cards = (tab?.cards || [])
      .filter((card) => {
        const type = card?.config?.type || "line";
        if (type !== "line") {
          console.warn(
            `[Project Layout] Dropping unsupported card type '${type}' in tab '${tab?.name ?? tabIndex}'`,
          );
          return false;
        }
        return true;
      })
      .map((card) => {
        const config = card.config || {};
        const yMetrics = Array.isArray(config.yMetrics)
          ? config.yMetrics.map(stripMetricSuffix).filter(Boolean)
          : [];
        const defaultMetrics = Array.isArray(card.defaultMetrics)
          ? card.defaultMetrics.map(stripMetricSuffix).filter(Boolean)
          : yMetrics.length === 1
            ? [yMetrics[0]]
            : [];
        return {
          id: card.id || `card-${nextCardId.value++}`,
          config: {
            type: "line",
            title:
              config.title || (yMetrics.length ? yMetrics.join(", ") : "Chart"),
            widthPercent: config.widthPercent || 33,
            height: config.height || 400,
            xMetric: config.xMetric || "global_step",
            yMetrics,
          },
          defaultMetrics,
        };
      });

    sanitized.push({
      name: tab?.name || `Tab ${tabIndex + 1}`,
      cards,
    });
  });

  return sanitized.length > 0 ? sanitized : [{ name: "Metrics", cards: [] }];
}

function createDefaultCard(metric, xMetric = "global_step") {
  return {
    id: `card-${nextCardId.value++}`,
    config: {
      type: "line",
      title: metric,
      widthPercent: 33,
      height: 400,
      xMetric,
      yMetrics: [metric],
    },
    defaultMetrics: [metric],
  };
}

function ensureDefaultCardsPresent() {
  if (!availableMetrics.value || availableMetrics.value.length === 0) {
    return false;
  }

  const existingMetrics = new Set();
  for (const tab of tabs.value) {
    for (const card of tab.cards || []) {
      if (card.config?.type !== "line") continue;
      for (const metric of card.config.yMetrics || []) {
        existingMetrics.add(stripMetricSuffix(metric));
      }
    }
  }

  const metricsToAdd = availableMetrics.value.filter((metric) => {
    if (AXIS_ONLY_METRICS.has(metric)) return false;
    if (removedMetrics.value.has(metric)) return false;
    return !existingMetrics.has(metric);
  });

  if (metricsToAdd.length === 0) {
    return false;
  }

  const namespaceMap = new Map();
  namespaceMap.set("", []);
  for (const metric of metricsToAdd) {
    const slashIdx = metric.indexOf("/");
    if (slashIdx > 0) {
      const namespace = metric.substring(0, slashIdx);
      if (!namespaceMap.has(namespace)) {
        namespaceMap.set(namespace, []);
      }
      namespaceMap.get(namespace).push(metric);
    } else {
      namespaceMap.get("").push(metric);
    }
  }

  if (tabs.value.length === 0) {
    tabs.value = [{ name: "Metrics", cards: [] }];
  }

  let metricsTab = tabs.value.find((t) => t.name === "Metrics");
  if (!metricsTab) {
    metricsTab = { name: "Metrics", cards: [] };
    tabs.value.unshift(metricsTab);
  }

  for (const metric of namespaceMap.get("") || []) {
    metricsTab.cards.push(createDefaultCard(metric, "global_step"));
  }

  for (const [namespace, metrics] of namespaceMap.entries()) {
    if (namespace === "" || metrics.length === 0) continue;
    let namespaceTab = tabs.value.find((t) => t.name === namespace);
    if (!namespaceTab) {
      namespaceTab = { name: namespace, cards: [] };
      tabs.value.push(namespaceTab);
    }
    for (const metric of metrics) {
      namespaceTab.cards.push(createDefaultCard(metric, "step"));
    }
  }

  tabs.value = tabs.value.map((tab) => ({
    ...tab,
    cards: [...tab.cards],
  }));
  return true;
}

// Build default tabs grouped by namespace
function buildDefaultTabs() {
  // Group metrics by namespace
  const metricsByNamespace = new Map();
  metricsByNamespace.set("", []); // Main namespace

  for (const metric of availableMetrics.value) {
    if (AXIS_ONLY_METRICS.has(metric)) continue;
    if (removedMetrics.value.has(metric)) continue;

    const slashIdx = metric.indexOf("/");
    if (slashIdx > 0) {
      const namespace = metric.substring(0, slashIdx);
      if (!metricsByNamespace.has(namespace)) {
        metricsByNamespace.set(namespace, []);
      }
      metricsByNamespace.get(namespace).push(metric);
    } else {
      metricsByNamespace.get("").push(metric);
    }
  }

  // Build tabs
  const newTabs = [];
  let cardId = 1;

  // Main tab
  const mainMetrics = metricsByNamespace.get("");
  if (mainMetrics.length > 0) {
    const cards = mainMetrics.map((metric) => {
      return {
        id: `card-${cardId++}`,
        config: {
          type: "line",
          title: metric,
          widthPercent: 33,
          height: 400,
          xMetric: "global_step",
          yMetrics: [metric], // Just the base metric name
        },
        defaultMetrics: [metric],
      };
    });
    newTabs.push({ name: "Metrics", cards });
  }

  // Namespace tabs
  for (const [namespace, metrics] of metricsByNamespace.entries()) {
    if (namespace !== "" && metrics.length > 0) {
      const cards = metrics.map((metric) => {
        return {
          id: `card-${cardId++}`,
          config: {
            type: "line",
            title: metric,
            widthPercent: 33,
            height: 400,
            xMetric: "step",
            yMetrics: [metric], // Just the base metric name
          },
          defaultMetrics: [metric],
        };
      });
      newTabs.push({ name: namespace, cards });
    }
  }

  tabs.value = newTabs.length > 0 ? newTabs : [{ name: "Metrics", cards: [] }];
  nextCardId.value = cardId;
}

// Fetch metrics for current tab (multi-run version)
async function fetchMetricsForTab(forceRefresh = false) {
  try {
    console.log(
      `[fetchMetricsForTab] Starting fetch, forceRefresh: ${forceRefresh}`,
    );
    const tab = tabs.value.find((t) => t.name === activeTab.value);
    if (!tab) return;

    const computedMetrics = new Set(["walltime", "relative_walltime"]);
    const neededMetrics = new Set();

    for (const card of tab.cards) {
      if (card.config.type === "line") {
        // Add xMetric (no run suffix)
        if (card.config.xMetric && !computedMetrics.has(card.config.xMetric)) {
          neededMetrics.add(card.config.xMetric);
        }
        // Add yMetrics (strip run suffix)
        if (card.config.yMetrics) {
          for (const yMetric of card.config.yMetrics) {
            if (!computedMetrics.has(yMetric)) {
              // Strip run name suffix: "metric (run_name)" -> "metric"
              const metricName = yMetric.includes(" (")
                ? yMetric.substring(0, yMetric.lastIndexOf(" ("))
                : yMetric;
              neededMetrics.add(metricName);
            }
          }
        }
      }
    }

    console.log(
      `[fetchMetricsForTab] Needed metrics for tab ${activeTab.value}:`,
      Array.from(neededMetrics),
    );

    if (neededMetrics.size === 0) {
      console.log("[fetchMetricsForTab] No metrics needed, returning");
      return;
    }

    // Fetch batch scalar data
    const runIds = displayedRuns.value.map((r) => r.run_id);
    console.log(`[fetchMetricsForTab] Fetching scalars for runs:`, runIds);
    console.log(
      `[fetchMetricsForTab] Requesting metrics:`,
      Array.from(neededMetrics),
    );

    const requestBody = {
      run_ids: runIds,
      metrics: Array.from(neededMetrics),
    };
    console.log(`[fetchMetricsForTab] Request body:`, requestBody);

    const response = await fetch(
      `/api/projects/${projectName.value}/runs/batch/scalars`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      },
    );

    if (!response.ok) {
      console.error(
        `[fetchMetricsForTab] Failed to fetch scalars: ${response.status}`,
      );
      throw new Error(`Failed to fetch scalars: ${response.status}`);
    }

    const batchData = await response.json();

    console.log("[fetchMetricsForTab] Batch data received:", batchData);
    // Log data sizes
    for (const [runId, runData] of Object.entries(batchData)) {
      console.log(
        `[fetchMetricsForTab] Run ${runId}:`,
        Object.keys(runData).length,
        "metrics",
      );
      for (const [metric, data] of Object.entries(runData)) {
        console.log(
          `[fetchMetricsForTab]   - ${metric}: ${data.steps?.length || 0} steps`,
        );
      }
    }

    // Convert to sparse data format for each run
    // Create new object to force reactivity
    const newCache = {};

    for (const run of displayedRuns.value) {
      const runId = run.run_id;
      const runData = batchData[runId];

      if (!runData) {
        console.warn(`No batch data for run ${runId}`);
        continue;
      }

      newCache[runId] = {};

      for (const [metric, data] of Object.entries(runData)) {
        console.log(`Processing ${runId}/${metric}:`, data);

        if (!data.steps || !data.values) {
          console.warn(`Invalid data structure for ${runId}/${metric}`);
          continue;
        }

        if (data.steps.length === 0) {
          console.warn(`Empty steps for ${runId}/${metric}`);
          // Store empty array for metrics with no data
          newCache[runId][metric] = [];
          continue;
        }

        const maxStep = Math.max(...data.steps);

        // Validate maxStep to avoid invalid array length
        if (!isFinite(maxStep) || maxStep < 0 || maxStep > 10000000) {
          console.warn(
            `Invalid maxStep ${maxStep} for ${metric} in ${runId}, skipping`,
          );
          continue;
        }

        const sparseArray = new Array(maxStep + 1).fill(null);

        for (let i = 0; i < data.steps.length; i++) {
          const step = data.steps[i];
          let value = data.values[i];

          // Convert special string markers
          if (value === "NaN") value = NaN;
          else if (value === "Infinity") value = Infinity;
          else if (value === "-Infinity") value = -Infinity;

          sparseArray[step] = value;
        }

        newCache[runId][metric] = sparseArray;
        console.log(
          `Stored ${runId}/${metric}: ${sparseArray.length} slots, ${sparseArray.filter((v) => v !== null).length} values`,
        );
      }
    }

    // Assign new object to trigger reactivity
    multiRunDataCache.value = newCache;
    console.log(
      "[fetchMetricsForTab] multiRunDataCache updated with",
      Object.keys(newCache).length,
      "runs",
    );
    console.log("[fetchMetricsForTab] Full cache:", multiRunDataCache.value);
    console.log("[fetchMetricsForTab] Fetch completed successfully");
  } catch (error) {
    console.error("[fetchMetricsForTab] Error:", error);
  }
}

// Build sparse data for ConfigurableChartCard (merge all runs with prefixes)
const sparseData = computed(() => {
  const data = {};

  // In multi-run mode, each run gets its OWN x-axis data paired with y-axis data
  // Format: "step (run_id)", "global_step (run_id)", "metric (run_id)"
  // This ensures each run's data is correctly aligned with its own step values

  for (const run of displayedRuns.value) {
    const runId = run.run_id;
    const runData = multiRunDataCache.value[runId];

    if (!runData) {
      // This is normal during initial render before data loads
      continue;
    }

    for (const [metric, values] of Object.entries(runData)) {
      // Add ALL metrics (including x-axis) with run suffix
      // This ensures each run has its own complete dataset
      const key = `${metric} (${runId})`;
      data[key] = values;

      console.log(
        `Added ${key}: ${values.length} slots, ${values.filter((v) => v !== null).length} non-null`,
      );
    }
  }

  console.log(`sparseData computed with ${Object.keys(data).length} series:`);
  console.log("sparseData keys:", Object.keys(data));
  console.log("sparseData full:", data);
  return data;
});

const currentTabCards = computed({
  get() {
    const tab = tabs.value.find((t) => t.name === activeTab.value);
    return tab ? tab.cards : [];
  },
  set(newCards) {
    const tab = tabs.value.find((t) => t.name === activeTab.value);
    if (tab) {
      tab.cards = newCards;
      saveLayout();
    }
  },
});

// Pagination (same as [id].vue)
const webglChartsPerPage = computed(() =>
  isMobile.value ? 2 : chartsPerPage.value,
);
const defaultCardHeight = computed(() => (isMobile.value ? 280 : 400));

const paginatedCards = computed(() => {
  const allCards = currentTabCards.value;
  let webglCount = 0;
  let pageCards = [];
  let currentPageCards = [];

  for (const card of allCards) {
    const isWebGL =
      card.config.type === "line" || card.config.type === "histogram";

    if (isWebGL) {
      webglCount++;
      if (webglCount > webglChartsPerPage.value) {
        pageCards.push(currentPageCards);
        currentPageCards = [card];
        webglCount = 1;
      } else {
        currentPageCards.push(card);
      }
    } else {
      currentPageCards.push(card);
    }
  }

  if (currentPageCards.length > 0) {
    pageCards.push(currentPageCards);
  }

  return pageCards;
});

const totalChartPages = computed(() => paginatedCards.value.length);

const visibleCards = computed(() => {
  if (paginatedCards.value.length === 0) return [];
  const page = Math.min(
    currentChartPage.value,
    paginatedCards.value.length - 1,
  );
  return paginatedCards.value[page] || [];
});

watch(activeTab, () => {
  if (isInitializing.value) return;
  currentChartPage.value = 0;
  fetchMetricsForTab();
  saveLayout();
});

function saveLayout() {
  // Save hidden runs (inverse of selected)
  const allRunIds = new Set(allRuns.value.map((r) => r.run_id));
  const hiddenRunIds = Array.from(allRunIds).filter(
    (runId) => !selectedRunIds.value.has(runId),
  );

  const layout = {
    tabs: tabs.value,
    activeTab: activeTab.value,
    nextCardId: nextCardId.value,
    globalSettings: globalSettings.value,
    customRunColors: customRunColors.value, // Save custom color mappings
    sidebarWidth: sidebarWidth.value, // Save sidebar width
    hiddenRunIds: hiddenRunIds, // Save which runs are hidden
    removedMetrics: Array.from(removedMetrics.value),
  };
  localStorage.setItem(storageKey.value, JSON.stringify(layout));
}

function calculateRows(cards) {
  const rows = [];
  let currentRow = [];
  let rowWidth = 0;

  for (let i = 0; i < cards.length; i++) {
    const w = cards[i].config.widthPercent || 100;
    if (rowWidth > 0 && rowWidth + w > 102) {
      rows.push([...currentRow]);
      currentRow = [];
      rowWidth = 0;
    }
    currentRow.push(i);
    rowWidth += w;
  }
  if (currentRow.length > 0) rows.push(currentRow);
  return rows;
}

function updateCard({ id, config, syncAll, realtime }) {
  if (!config) return;

  const currentTab = tabs.value.find((t) => t.name === activeTab.value);
  if (!currentTab) return;

  const cardIndex = currentTab.cards.findIndex((c) => c.id === id);
  if (cardIndex === -1) return;

  const oldConfig = currentTab.cards[cardIndex].config;
  const heightChanged = oldConfig.height !== config.height;
  const widthChanged = oldConfig.widthPercent !== config.widthPercent;

  if (syncAll) {
    if (realtime && isUpdating.value) return;

    if (!realtime) {
      if (isUpdating.value) return;
      isUpdating.value = true;
    }

    try {
      for (let i = 0; i < currentTab.cards.length; i++) {
        const updates = {};
        if (heightChanged) updates.height = config.height;
        if (widthChanged) updates.widthPercent = config.widthPercent;

        if (Object.keys(updates).length > 0) {
          currentTab.cards[i].config = {
            ...currentTab.cards[i].config,
            ...updates,
          };
        }
      }
    } finally {
      if (!realtime) {
        nextTick(() => {
          isUpdating.value = false;
          saveLayout();
        });
      }
    }
    return;
  }

  currentTab.cards[cardIndex].config = config;

  if (isUpdating.value) return;
  isUpdating.value = true;

  try {
    if (heightChanged && !widthChanged) {
      const rows = calculateRows(currentTab.cards);
      const currentRow = rows.find((row) => row.includes(cardIndex));

      if (currentRow && currentRow.length > 1) {
        for (const idx of currentRow) {
          if (idx !== cardIndex) {
            currentTab.cards[idx].config = {
              ...currentTab.cards[idx].config,
              height: config.height,
            };
          }
        }
      }
    } else if (widthChanged) {
      const rows = calculateRows(currentTab.cards);

      for (const row of rows) {
        if (row.length > 1) {
          const isActiveCardInRow = row.includes(cardIndex);
          let targetHeight = null;

          const otherCards = row.filter((idx) => idx !== cardIndex);
          if (otherCards.length > 0) {
            targetHeight = currentTab.cards[otherCards[0]].config.height;
          } else if (isActiveCardInRow) {
            targetHeight = currentTab.cards[cardIndex].config.height;
          }

          if (targetHeight !== null) {
            for (const idx of row) {
              if (currentTab.cards[idx].config.height !== targetHeight) {
                currentTab.cards[idx].config = {
                  ...currentTab.cards[idx].config,
                  height: targetHeight,
                };
              }
            }
          }
        }
      }
    }
  } finally {
    nextTick(() => {
      isUpdating.value = false;
      saveLayout();
    });
  }
}

function removeCard(id) {
  const tab = tabs.value.find((t) => t.name === activeTab.value);
  if (tab) {
    const cardToRemove = tab.cards.find((c) => c.id === id);
    if (cardToRemove) {
      const defaultList =
        cardToRemove.defaultMetrics && cardToRemove.defaultMetrics.length > 0
          ? cardToRemove.defaultMetrics
          : cardToRemove.config?.yMetrics?.length === 1
            ? cardToRemove.config.yMetrics
            : [];
      defaultList.forEach((metric) => {
        const baseMetric = stripMetricSuffix(metric);
        if (baseMetric) {
          removedMetrics.value.add(baseMetric);
        }
      });
    }

    tab.cards = tab.cards.filter((c) => c.id !== id);
    saveLayout();
  }
}

function onDragEnd(evt) {
  const currentTab = tabs.value.find((t) => t.name === activeTab.value);
  if (!currentTab) return;

  const draggedIndex = evt.newIndex;
  const oldIndex = evt.oldIndex;

  if (draggedIndex === undefined || oldIndex === undefined) {
    saveLayout();
    return;
  }

  const newRows = calculateRows(currentTab.cards);

  for (const row of newRows) {
    if (row.length > 1) {
      const heightCounts = {};

      for (const idx of row) {
        const h = currentTab.cards[idx].config.height;
        heightCounts[h] = (heightCounts[h] || 0) + 1;
      }

      let dominantHeight = null;
      let maxCount = 0;

      for (const [height, count] of Object.entries(heightCounts)) {
        if (count > maxCount) {
          maxCount = count;
          dominantHeight = parseInt(height);
        }
      }

      if (dominantHeight !== null) {
        for (const idx of row) {
          if (currentTab.cards[idx].config.height !== dominantHeight) {
            currentTab.cards[idx].config = {
              ...currentTab.cards[idx].config,
              height: dominantHeight,
            };
          }
        }
      }
    }
  }

  saveLayout();
}

function applyGlobalSettings() {
  const tabIndex = tabs.value.findIndex((t) => t.name === activeTab.value);
  if (tabIndex === -1) return;

  const newCards = tabs.value[tabIndex].cards.map((card) => {
    const newConfig = { ...card.config };

    if (card.config.type === "line") {
      newConfig.xMetric = globalSettings.value.xAxis;
      newConfig.smoothingMode = globalSettings.value.smoothing;
      newConfig.smoothingValue = globalSettings.value.smoothingValue;
      newConfig.downsampleRate = globalSettings.value.downsampleRate;
    }

    return { ...card, config: newConfig };
  });

  tabs.value = [
    ...tabs.value.slice(0, tabIndex),
    { ...tabs.value[tabIndex], cards: newCards },
    ...tabs.value.slice(tabIndex + 1),
  ];

  nextTick(() => {
    saveLayout();
    showGlobalSettings.value = false;
    ElMessage.success("Applied global settings to all cards");
  });
}

function addCard() {
  showAddChartDialog.value = true;
  newChartType.value = "line";
  newChartValue.value = [];
}

const availableChartValues = computed(() => {
  return availableMetrics.value.filter((m) => !AXIS_ONLY_METRICS.has(m));
});

function confirmAddChart() {
  const tab = tabs.value.find((t) => t.name === activeTab.value);
  if (!tab) return;

  if (!Array.isArray(newChartValue.value) || newChartValue.value.length === 0) {
    ElMessage.warning("Please select at least one metric");
    return;
  }

  const title = newChartValue.value.join(", ");

  const newCard = {
    id: `card-${nextCardId.value++}`,
    config: {
      type: "line",
      title: title,
      widthPercent: 33,
      height: 400,
      xMetric: "global_step",
      yMetrics: newChartValue.value,
    },
    defaultMetrics: [],
  };

  tab.cards.push(newCard);
  saveLayout();
  showAddChartDialog.value = false;
  newChartValue.value = [];

  // Fetch metrics for the new chart
  fetchMetricsForTab();
}

function resetLayout() {
  if (
    confirm("Reset to default layout? This will remove all customizations.")
  ) {
    removedMetrics.value = new Set();
    localStorage.removeItem(storageKey.value);
    location.reload();
  }
}

function toggleSidebar() {
  isSidebarCollapsed.value = !isSidebarCollapsed.value;
}

function startResizeSidebar(e) {
  if (isMobile.value) return; // No resize on mobile

  e.preventDefault();
  isResizingSidebar.value = true;
  const startX = e.clientX;
  const startWidth = sidebarWidth.value;

  const onMove = (e) => {
    const delta = e.clientX - startX;
    const newWidth = Math.max(
      minSidebarWidth,
      Math.min(maxSidebarWidth, startWidth + delta),
    );
    sidebarWidth.value = newWidth;
  };

  const onUp = () => {
    isResizingSidebar.value = false;
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    // Save sidebar width
    saveLayout();
  };

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

// Provide sidebar state to TheHeader via global window object (simple approach)
watch(
  [isMobile, isSidebarCollapsed],
  () => {
    if (isMobile.value) {
      window.__projectSidebarState = {
        showToggle: true,
        collapsed: isSidebarCollapsed.value,
        toggle: toggleSidebar,
      };
    } else {
      window.__projectSidebarState = null;
    }
  },
  { immediate: true },
);

onMounted(() => {
  fetchRuns();

  // Start polling for new runs every 5 seconds
  pollInterval = setInterval(() => {
    pollRuns();
  }, 5000);
});

onUnmounted(() => {
  window.__projectSidebarState = null;

  // Cleanup polling
  if (pollInterval) {
    clearInterval(pollInterval);
  }
});
</script>

<template>
  <div class="project-comparison flex h-screen overflow-hidden">
    <!-- Backdrop overlay (mobile only, when sidebar open) -->
    <div
      v-if="isMobile && !isSidebarCollapsed"
      @click="toggleSidebar"
      class="sidebar-backdrop"
    />

    <!-- Left Sidebar -->
    <aside
      v-show="!isMobile || !isSidebarCollapsed"
      :class="{ 'sidebar-mobile': isMobile, resizing: isResizingSidebar }"
      :style="{ width: isMobile ? '100%' : `${sidebarWidth}px` }"
      class="border-r border-gray-200 dark:border-gray-700 flex flex-col bg-white dark:bg-gray-900 relative"
    >
      <!-- Header -->
      <div class="p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 class="text-lg font-bold mb-2">Runs ({{ totalRuns }})</h3>
        <div class="flex gap-2">
          <el-button size="small" @click="selectAllOnPage" class="flex-1">
            Select Page
          </el-button>
          <el-button size="small" @click="deselectAll" class="flex-1">
            Clear
          </el-button>
        </div>
      </div>

      <!-- Warning if > 10 selected -->
      <el-alert
        v-if="selectedRunIds.size > 10"
        type="warning"
        :closable="false"
        class="m-2"
      >
        <template #title>{{ selectedRunIds.size }} runs selected</template>
        Only showing last 10 in charts for performance.
      </el-alert>

      <!-- Run list -->
      <div class="flex-1 overflow-y-auto">
        <div v-if="loading" class="p-4 text-center">
          <el-icon class="is-loading"><Loading /></el-icon>
          <p class="mt-2 text-sm">Loading runs...</p>
        </div>

        <RunSelectionList
          v-else
          :runs="paginatedRuns"
          :selected-run-ids="selectedRunIds"
          :displayed-run-ids="displayedRuns.map((r) => r.run_id)"
          :run-colors="runColors"
          :project="projectName"
          @toggle="toggleRunSelection"
          @update-color="updateRunColor"
        />
      </div>

      <!-- Pagination -->
      <div class="p-4 border-t border-gray-200 dark:border-gray-700">
        <el-pagination
          :total="totalRuns"
          :page-size="runsPerPage"
          :current-page="currentPage"
          layout="prev, pager, next"
          small
          @current-change="handlePageChange"
        />
      </div>

      <!-- Resize handle (desktop only) -->
      <div
        v-if="!isMobile"
        @mousedown="startResizeSidebar"
        class="sidebar-resize-handle"
        title="Drag to resize sidebar"
      />
    </aside>

    <!-- Right Main Area -->
    <main
      class="flex-1 flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-800"
    >
      <!-- Top bar -->
      <div
        class="p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 flex flex-wrap justify-between items-start gap-4"
      >
        <div>
          <h2 class="text-xl font-bold">{{ projectName }}</h2>
          <p class="text-sm text-gray-500 dark:text-gray-400">
            Comparing {{ displayedRuns.length }} runs
          </p>
        </div>

        <div class="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-600 dark:text-gray-400"
              >Legend labels:</span
            >
            <el-radio-group v-model="legendMode" size="small">
              <el-radio-button value="annotation">Annotation</el-radio-button>
              <el-radio-button value="name">Run Name</el-radio-button>
            </el-radio-group>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-600 dark:text-gray-400"
              >Charts per page:</span
            >
            <el-radio-group v-model="chartsPerPage" size="small">
              <el-radio-button :value="6">6</el-radio-button>
              <el-radio-button :value="8">8</el-radio-button>
              <el-radio-button :value="12">12</el-radio-button>
            </el-radio-group>
          </div>
        </div>
      </div>

      <!-- Chart area (same structure as [id].vue) -->
      <div class="flex-1 overflow-y-auto p-4">
        <!-- Action buttons -->
        <div class="mb-4 flex items-center justify-end gap-2">
          <el-button type="primary" size="small" @click="addCard">
            <i class="i-ep-plus mr-1" />
            Add Chart
          </el-button>
          <el-button size="small" @click="resetLayout" type="danger" plain>
            <i class="i-ep-refresh-left mr-1" />
            Reset Layout
          </el-button>
        </div>

        <el-tabs v-model="activeTab" type="card">
          <el-tab-pane
            v-for="tab in tabs"
            :key="tab.name"
            :label="tab.name"
            :name="tab.name"
          >
            <template #label>
              <div class="flex items-center gap-2">
                <span>{{ tab.name }}</span>
                <el-button
                  v-if="tab.name === activeTab && !isEditingTabs"
                  size="small"
                  circle
                  @click.stop="toggleHoverSync"
                  :title="
                    hoverSyncEnabled
                      ? 'Disable Hover Sync'
                      : 'Enable Hover Sync'
                  "
                  :type="hoverSyncEnabled ? 'primary' : 'default'"
                >
                  <i class="i-ep-connection"></i>
                </el-button>
                <el-button
                  v-if="tab.name === activeTab && !isEditingTabs"
                  size="small"
                  circle
                  @click.stop="showGlobalSettings = true"
                  title="Global Tab Settings"
                >
                  <i class="i-ep-setting"></i>
                </el-button>
              </div>
            </template>
          </el-tab-pane>
        </el-tabs>

        <!-- Pagination controls -->
        <div
          v-if="totalChartPages > 1"
          class="flex items-center justify-center gap-4 mb-4"
        >
          <el-button
            size="small"
            :disabled="currentChartPage === 0"
            @click="currentChartPage--"
            icon="ArrowLeft"
          >
            Previous
          </el-button>
          <span class="text-sm text-gray-600 dark:text-gray-400">
            Page {{ currentChartPage + 1 }} / {{ totalChartPages }}
            <span class="text-xs ml-2">
              ({{ visibleCards.length }} charts, max
              {{ webglChartsPerPage }} WebGL per page)
            </span>
          </span>
          <el-button
            size="small"
            :disabled="currentChartPage >= totalChartPages - 1"
            @click="currentChartPage++"
            icon="ArrowRight"
          >
            Next
          </el-button>
        </div>

        <!-- Charts using ConfigurableChartCard -->
        <VueDraggable
          v-model="currentTabCards"
          :animation="animationsEnabled ? 200 : 0"
          handle=".card-drag-handle"
          class="flex flex-wrap gap-4"
          @end="onDragEnd"
        >
          <ConfigurableChartCard
            v-for="card in visibleCards"
            :key="card.id"
            :card-id="card.id"
            :experiment-id="displayedRuns[0]?.run_id"
            :project="projectName"
            :sparse-data="sparseData"
            :available-metrics="availableMetrics"
            :initial-config="{
              ...card.config,
              height: isMobile ? defaultCardHeight : card.config.height,
            }"
            :tab-name="activeTab"
            :hover-sync-enabled="hoverSyncEnabled"
            :multi-run-mode="true"
            :run-colors="runColors"
            :run-names="runNames"
            @update:config="updateCard"
            @remove="removeCard(card.id)"
          />
        </VueDraggable>

        <el-empty
          v-if="currentTabCards.length === 0"
          description="No charts in this tab"
        />
      </div>
    </main>

    <!-- Add Chart Dialog -->
    <el-dialog v-model="showAddChartDialog" title="Add Chart" width="500px">
      <el-form label-width="120px">
        <el-form-item label="Select Metrics">
          <el-select
            v-model="newChartValue"
            class="w-full"
            placeholder="Choose one or more metrics"
            multiple
            filterable
            collapse-tags
            collapse-tags-tooltip
          >
            <el-option
              v-for="metric in availableChartValues"
              :key="metric"
              :label="metric"
              :value="metric"
            />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddChartDialog = false">Cancel</el-button>
        <el-button type="primary" @click="confirmAddChart">Add</el-button>
      </template>
    </el-dialog>

    <!-- Global Settings Dialog -->
    <el-dialog
      v-model="showGlobalSettings"
      title="Global Tab Settings"
      width="600px"
    >
      <el-form label-width="160px">
        <el-divider content-position="left">Line Chart Settings</el-divider>

        <el-form-item label="X-Axis">
          <el-select v-model="globalSettings.xAxis" class="w-full">
            <el-option
              v-for="metric in availableMetrics"
              :key="metric"
              :label="metric"
              :value="metric"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="Smoothing">
          <el-select v-model="globalSettings.smoothing" class="w-full">
            <el-option label="Disabled" value="disabled" />
            <el-option label="EMA" value="ema" />
            <el-option label="Moving Average" value="ma" />
            <el-option label="Gaussian" value="gaussian" />
          </el-select>
        </el-form-item>

        <el-form-item
          v-if="globalSettings.smoothing !== 'disabled'"
          label="Smoothing Value"
        >
          <el-input-number
            v-model="globalSettings.smoothingValue"
            :min="0"
            :max="globalSettings.smoothing === 'ema' ? 1 : 1000"
            :step="globalSettings.smoothing === 'ema' ? 0.01 : 1"
            class="w-full"
          />
        </el-form-item>

        <el-form-item label="Downsample Rate">
          <el-input-number
            v-model="globalSettings.downsampleRate"
            :min="-1"
            :max="100"
            class="w-full"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="showGlobalSettings = false">Cancel</el-button>
        <el-button type="primary" @click="applyGlobalSettings">
          Apply to All Cards
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.project-comparison {
  height: 100vh;
  max-height: 100vh;
}

.sidebar-resize-handle {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 6px;
  cursor: ew-resize;
  background: transparent;
  transition: background 0.2s;
  z-index: 10;
}

.sidebar-resize-handle:hover {
  background: rgba(64, 158, 255, 0.3);
}

.resizing {
  user-select: none;
}

.sidebar-mobile {
  position: fixed;
  top: 57px; /* Below header (header height ~57px) */
  left: 0;
  bottom: 0;
  z-index: 40;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
}

/* Backdrop overlay when sidebar is open on mobile */
.sidebar-backdrop {
  position: fixed;
  top: 57px;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 39;
}

@media (max-width: 900px) {
  aside {
    width: 80% !important;
    max-width: 400px !important;
  }
}
</style>
