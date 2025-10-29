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
 * Strategy: Use run index in sorted list + run_id hash jittering
 *
 * - Base color from index (stable as long as run exists)
 * - Slight HSL jitter based on run_id hash (makes each run unique)
 * - Same run_id always gets same jittered color
 */
function getRunColor(runId, runIndex) {
  // Get base color from palette using index
  const baseColor = RUN_COLORS[runIndex % RUN_COLORS.length];

  // Generate deterministic jitter from run_id hash
  let hash = 0;
  for (let i = 0; i < runId.length; i++) {
    const char = runId.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }

  // Convert hash to jitter values (-15 to +15 for hue, -10% to +10% for saturation/lightness)
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

const storageKey = computed(() => `project-layout-${route.params.project}`);

// Custom run colors (saved by user, overrides hash-based defaults)
// Now saved with layout instead of separate storage
const customRunColors = ref({});

// Run colors map - stable index + deterministic jittering
// Map by both run_id AND run name for easy lookup
const runColors = computed(() => {
  const colors = {};
  allRuns.value.forEach((run, index) => {
    // Use custom color if set, otherwise use index + hash jittering
    const color =
      customRunColors.value[run.run_id] || getRunColor(run.run_id, index);

    colors[run.run_id] = color;
    // Also map by name for LinePlot lookup
    if (run.name) {
      colors[run.name] = color;
    }
    // Also map by the display name we use (name or run_id)
    colors[run.name || run.run_id] = color;
  });
  return colors;
});

// Run names map - map run_id to run name for display
const runNames = computed(() => {
  const names = {};
  allRuns.value.forEach((run) => {
    names[run.run_id] = run.name || run.run_id;
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

    // Default: select all runs (use run_id field)
    selectedRunIds.value = new Set(allRuns.value.map((r) => r.run_id));

    // Initialize project view
    await initializeProject();
  } catch (error) {
    console.error("Failed to fetch runs:", error);
    ElMessage.error("Failed to fetch project runs");
  } finally {
    loading.value = false;
  }
}

// Poll for new runs in background
async function pollRuns() {
  try {
    const response = await fetch(`/api/projects/${projectName.value}/runs`);
    if (!response.ok) return;

    const data = await response.json();
    const newRuns = (data.runs || []).sort((a, b) => {
      const dateA = new Date(a.created_at || 0);
      const dateB = new Date(b.created_at || 0);
      return dateB - dateA;
    });

    // Check if there are new runs
    const oldRunIds = new Set(allRuns.value.map((r) => r.run_id));
    const hasNewRuns = newRuns.some((r) => !oldRunIds.has(r.run_id));

    if (hasNewRuns) {
      console.log("New runs detected, updating list");
      allRuns.value = newRuns;
      // Auto-select new runs
      selectedRunIds.value = new Set(allRuns.value.map((r) => r.run_id));
    }
  } catch (error) {
    console.error("Failed to poll runs:", error);
  }
}

function toggleRunSelection(runId) {
  if (selectedRunIds.value.has(runId)) {
    selectedRunIds.value.delete(runId);
  } else {
    selectedRunIds.value.add(runId);
  }

  // Trigger reactivity
  selectedRunIds.value = new Set(selectedRunIds.value);
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
    const saved = localStorage.getItem(storageKey.value);
    let savedLayout = null;

    if (saved) {
      savedLayout = JSON.parse(saved);
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
    }

    // Build default cards if no saved layout
    if (savedLayout?.tabs) {
      tabs.value = savedLayout.tabs;
    } else {
      buildDefaultTabs();
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

        console.log(
          `Updated card ${card.id} yMetrics to base metrics only:`,
          card.config.yMetrics,
        );
      }
    }
  }
}

// Build default tabs grouped by namespace
function buildDefaultTabs() {
  const axisOnlyMetrics = new Set([
    "step",
    "global_step",
    "timestamp",
    "walltime",
    "relative_walltime",
  ]);

  // Group metrics by namespace
  const metricsByNamespace = new Map();
  metricsByNamespace.set("", []); // Main namespace

  for (const metric of availableMetrics.value) {
    if (axisOnlyMetrics.has(metric)) continue;

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
        };
      });
      newTabs.push({ name: namespace, cards });
    }
  }

  tabs.value = newTabs.length > 0 ? newTabs : [{ name: "Metrics", cards: [] }];
  nextCardId.value = cardId;
}

// Fetch metrics for current tab (multi-run version)
async function fetchMetricsForTab() {
  try {
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
      `Needed metrics for tab ${activeTab.value}:`,
      Array.from(neededMetrics),
    );

    if (neededMetrics.size === 0) return;

    // Fetch batch scalar data
    const runIds = displayedRuns.value.map((r) => r.run_id);
    const response = await fetch(
      `/api/projects/${projectName.value}/runs/batch/scalars`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_ids: runIds,
          metrics: Array.from(neededMetrics),
        }),
      },
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch scalars: ${response.status}`);
    }

    const batchData = await response.json();

    console.log("Batch data received:", batchData);

    // Convert to sparse data format for each run
    multiRunDataCache.value = {};

    for (const run of displayedRuns.value) {
      const runId = run.run_id;
      const runData = batchData[runId];

      if (!runData) {
        console.warn(`No batch data for run ${runId}`);
        continue;
      }

      multiRunDataCache.value[runId] = {};

      for (const [metric, data] of Object.entries(runData)) {
        console.log(`Processing ${runId}/${metric}:`, data);

        if (!data.steps || !data.values) {
          console.warn(`Invalid data structure for ${runId}/${metric}`);
          continue;
        }

        if (data.steps.length === 0) {
          console.warn(`Empty steps for ${runId}/${metric}`);
          // Store empty array for metrics with no data
          multiRunDataCache.value[runId][metric] = [];
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

        multiRunDataCache.value[runId][metric] = sparseArray;
        console.log(
          `Stored ${runId}/${metric}: ${sparseArray.length} slots, ${sparseArray.filter((v) => v !== null).length} values`,
        );
      }
    }

    console.log("multiRunDataCache:", multiRunDataCache.value);
  } catch (error) {
    console.error("Failed to fetch metrics:", error);
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
  const layout = {
    tabs: tabs.value,
    activeTab: activeTab.value,
    nextCardId: nextCardId.value,
    globalSettings: globalSettings.value,
    customRunColors: customRunColors.value, // Save custom color mappings
    sidebarWidth: sidebarWidth.value, // Save sidebar width
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
  const axisOnlyMetrics = new Set([
    "step",
    "global_step",
    "timestamp",
    "walltime",
    "relative_walltime",
  ]);
  return availableMetrics.value.filter((m) => !axisOnlyMetrics.has(m));
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
        class="p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 flex justify-between items-center"
      >
        <div>
          <h2 class="text-xl font-bold">{{ projectName }}</h2>
          <p class="text-sm text-gray-500 dark:text-gray-400">
            Comparing {{ displayedRuns.length }} runs
          </p>
        </div>

        <div class="flex items-center gap-4">
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
