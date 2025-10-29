<script setup>
import { ElMessage } from "element-plus";
import { VueDraggable } from "vue-draggable-plus";
import ConfigurableChartCard from "@/components/ConfigurableChartCard.vue";
import { useAnimationPreference } from "@/composables/useAnimationPreference";
import { useHoverSync } from "@/composables/useHoverSync";

const route = useRoute();
const { animationsEnabled } = useAnimationPreference();
const { hoverSyncEnabled, toggleHoverSync } = useHoverSync();
const metricDataCache = ref({});
const availableMetrics = ref([]);
const availableSummary = ref(null); // Store full summary data

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
const newChartValue = ref([]); // Array for line charts, string for others
const isInitializing = ref(true); // Prevent watch triggers during init
const removedMetrics = ref(new Set()); // Track explicitly removed metrics

// Polling state
const lastMetricsCount = ref(0);
const pollingInterval = ref(null);
const pollingEnabled = ref(true);

// Pagination for WebGL context limit
const currentPage = ref(0);
const isMobile = ref(window.innerWidth <= 900); // Match CSS breakpoint

// Realtime responsive detection
const mediaQuery = window.matchMedia("(max-width: 900px)");
const handleResize = (e) => {
  isMobile.value = e.matches;
  currentPage.value = 0; // Reset to first page when switching mobile/desktop
};

onMounted(() => {
  mediaQuery.addEventListener("change", handleResize);
});

onUnmounted(() => {
  mediaQuery.removeEventListener("change", handleResize);
});

// Global settings (apply to all cards in current tab)
const globalSettings = ref({
  xAxis: "global_step",
  smoothing: "disabled",
  smoothingValue: 0.9,
  histogramMode: "flow",
  downsampleRate: -1, // -1 = adaptive
});

const storageKey = computed(
  () => `experiment-layout-${route.params.project}-${route.params.id}`,
);

// Extracted initialization logic for reuse
async function initializeExperiment() {
  try {
    isInitializing.value = true;

    const projectName = route.params.project;
    const runId = route.params.id;

    console.log(`[Init] Loading experiment: ${projectName}/${runId}`);

    // Clear previous data
    metricDataCache.value = {};

    // Fetch summary using runs API
    const summaryResponse = await fetch(
      `/api/projects/${projectName}/runs/${runId}/summary`,
    );

    if (!summaryResponse.ok) {
      console.error(
        "Failed to fetch summary:",
        summaryResponse.status,
        summaryResponse.statusText,
      );
      const errorText = await summaryResponse.text();
      console.error("Error response:", errorText);
      throw new Error(`Failed to fetch summary: ${summaryResponse.status}`);
    }

    const summary = await summaryResponse.json();
    console.log("Summary response:", summary);

    if (!summary.available_data || !summary.available_data.scalars) {
      console.error("Invalid summary structure:", summary);
      throw new Error("Invalid summary response structure");
    }

    // Store full summary for later use (media, tables, histograms)
    availableSummary.value = summary;
    availableMetrics.value = summary.available_data.scalars;

    // Add computed time metrics (will be calculated after fetching timestamp)
    // These are not in the backend, but calculated on frontend
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

      // Load global settings if saved
      if (savedLayout.globalSettings) {
        globalSettings.value = savedLayout.globalSettings;
      }

      // Load removed metrics list
      if (savedLayout.removedMetrics) {
        removedMetrics.value = new Set(savedLayout.removedMetrics);
      }
    }

    // Build default cards for new metrics (not in saved layout and not explicitly removed)
    const savedMetricSet = new Set();
    if (savedLayout?.tabs) {
      // Collect all metrics from saved layout
      for (const tab of savedLayout.tabs) {
        for (const card of tab.cards) {
          if (card.config.yMetrics) {
            card.config.yMetrics.forEach((m) => savedMetricSet.add(m));
          }
        }
      }
    }

    // Determine which metrics need default cards
    const needsDefaultCard = (metric) => {
      return (
        !savedMetricSet.has(metric) && // Not in saved layout
        !removedMetrics.value.has(metric) // Not explicitly removed
      );
    };

    // If we have saved layout, use it as base and add new metrics
    if (savedLayout?.tabs) {
      tabs.value = savedLayout.tabs;

      // Add default cards for new metrics to appropriate tabs
      let cardId = nextCardId.value;
      // Metrics that should NOT have default charts (only for axis selection)
      const axisOnlyMetrics = new Set([
        "step",
        "global_step",
        "timestamp",
        "walltime",
        "relative_walltime",
      ]);

      // Group NEW metrics by namespace
      const metricsByNamespace = new Map();
      metricsByNamespace.set("", []); // Main namespace

      for (const metric of availableMetrics.value) {
        // Skip axis-only metrics and metrics that already have cards or were removed
        if (axisOnlyMetrics.has(metric) || !needsDefaultCard(metric)) {
          continue;
        }

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

      // Add new metric cards to main tab
      const mainTab = tabs.value.find((t) => t.name === "Metrics");
      if (mainTab) {
        for (const metric of metricsByNamespace.get("")) {
          mainTab.cards.push({
            id: `card-${cardId++}`,
            config: {
              type: "line",
              title: metric,
              widthPercent: 33,
              height: 400,
              xMetric: "global_step",
              yMetrics: [metric],
            },
          });
        }
      }

      // Add new namespaced metric cards (create tabs if needed)
      for (const [namespace, metrics] of metricsByNamespace.entries()) {
        if (namespace === "") continue; // Already handled

        // Check if tab exists
        let namespaceTab = tabs.value.find((t) => t.name === namespace);
        if (!namespaceTab) {
          // Create new tab for this namespace
          namespaceTab = { name: namespace, cards: [] };
          tabs.value.push(namespaceTab);
          console.log(`[Init] Created new tab for namespace: ${namespace}`);
        }

        // Add cards for new metrics in this namespace
        for (const metric of metrics) {
          namespaceTab.cards.push({
            id: `card-${cardId++}`,
            config: {
              type: "line",
              title: metric,
              widthPercent: 33,
              height: 400,
              xMetric: "global_step",
              yMetrics: [metric],
            },
          });
        }
      }

      nextCardId.value = cardId;
    } else {
      // No saved layout - create default layout
      const cards = [];
      let cardId = 1;

      // Metrics that should NOT have default charts (only for axis selection)
      const axisOnlyMetrics = new Set([
        "step",
        "global_step",
        "timestamp",
        "walltime",
        "relative_walltime",
      ]);

      // Group metrics by namespace (before "/" is namespace)
      const metricsByNamespace = new Map();
      metricsByNamespace.set("", []); // Main namespace

      for (const metric of availableMetrics.value) {
        // Skip axis-only metrics - they shouldn't get default charts
        if (axisOnlyMetrics.has(metric)) {
          continue;
        }

        const slashIdx = metric.indexOf("/");
        if (slashIdx > 0) {
          // Has namespace: "train/loss" -> namespace="train", name="loss"
          const namespace = metric.substring(0, slashIdx);
          if (!metricsByNamespace.has(namespace)) {
            metricsByNamespace.set(namespace, []);
          }
          metricsByNamespace.get(namespace).push(metric);
        } else {
          // No namespace, goes to main
          metricsByNamespace.get("").push(metric);
        }
      }

      // Create cards for main namespace
      for (const metric of metricsByNamespace.get("")) {
        cards.push({
          id: `card-${cardId++}`,
          config: {
            type: "line",
            title: metric,
            widthPercent: 33,
            height: 400,
            xMetric: "global_step",
            yMetrics: [metric],
          },
        });
      }

      tabs.value[0].cards = cards;

      // Create tabs for each namespace (scalars + tables + histograms)
      const tabsByNamespace = new Map();

      // Add scalar metrics to tabs
      for (const [namespace, metrics] of metricsByNamespace.entries()) {
        if (namespace !== "") {
          if (!tabsByNamespace.has(namespace)) {
            tabsByNamespace.set(namespace, []);
          }
          for (const metric of metrics) {
            tabsByNamespace.get(namespace).push({
              id: `card-${cardId++}`,
              config: {
                type: "line",
                title: metric,
                widthPercent: 33,
                height: 400,
                xMetric: "step",
                yMetrics: [metric],
              },
            });
          }
        }
      }

      // Add tables to namespace tabs
      for (const tableName of summary.available_data?.tables || []) {
        const slashIdx = tableName.indexOf("/");
        if (slashIdx > 0) {
          const namespace = tableName.substring(0, slashIdx);
          if (!tabsByNamespace.has(namespace)) {
            tabsByNamespace.set(namespace, []);
          }
          tabsByNamespace.get(namespace).push({
            id: `card-${cardId++}`,
            config: {
              type: "table",
              title: tableName,
              widthPercent: 50,
              height: 400,
              tableName: tableName,
              currentStep: 0,
            },
          });
        }
      }

      // Add histograms to namespace tabs
      for (const histName of summary.available_data?.histograms || []) {
        const slashIdx = histName.indexOf("/");
        if (slashIdx > 0) {
          const namespace = histName.substring(0, slashIdx);
          if (!tabsByNamespace.has(namespace)) {
            tabsByNamespace.set(namespace, []);
          }

          // Default to flow mode for gradients/params
          const defaultMode =
            namespace === "gradients" || namespace === "params"
              ? "flow"
              : "single";

          tabsByNamespace.get(namespace).push({
            id: `card-${cardId++}`,
            config: {
              type: "histogram",
              title: histName,
              widthPercent: 33,
              height: 400,
              histogramName: histName,
              currentStep: 0,
              histogramMode: defaultMode, // Add mode to config
            },
          });
        }
      }

      // Create tabs from collected cards
      for (const [namespace, namespaceCards] of tabsByNamespace.entries()) {
        if (namespaceCards.length > 0) {
          tabs.value.push({ name: namespace, cards: namespaceCards });
        }
      }

      // Media/tables/histograms without namespace go to main tab
      for (const mediaName of summary.available_data?.media || []) {
        if (!mediaName.includes("/")) {
          cards.push({
            id: `card-${cardId++}`,
            config: {
              type: "media",
              title: mediaName,
              widthPercent: 33,
              height: 400,
              mediaName: mediaName,
              currentStep: 0,
            },
          });
        }
      }

      for (const tableName of summary.available_data?.tables || []) {
        if (!tableName.includes("/")) {
          cards.push({
            id: `card-${cardId++}`,
            config: {
              type: "table",
              title: tableName,
              widthPercent: 50,
              height: 400,
              tableName: tableName,
              currentStep: 0,
            },
          });
        }
      }

      for (const histName of summary.available_data?.histograms || []) {
        if (!histName.includes("/")) {
          cards.push({
            id: `card-${cardId++}`,
            config: {
              type: "histogram",
              title: histName,
              widthPercent: 33,
              height: 400,
              histogramName: histName,
              currentStep: 0,
            },
          });
        }
      }

      tabs.value[0].cards = cards;
      nextCardId.value = cardId;
    }

    // Fetch all metrics needed by visible cards
    await fetchMetricsForTab();

    // Determine default x-axis (prefer global_step if it's used, otherwise step)
    await determineDefaultXAxis();

    // Initialization complete - allow watch to fire on user tab changes
    isInitializing.value = false;
    console.log("[Init] Initialization complete");

    // Start polling for updates (every 3 seconds)
    startPolling();
  } catch (error) {
    console.error("Failed to load experiment:", error);
    isInitializing.value = false;
  }
}

onMounted(() => {
  initializeExperiment();
});

onUnmounted(() => {
  // Stop polling when leaving page
  stopPolling();
});

async function determineDefaultXAxis() {
  // Check if global_step has any non-zero values
  try {
    const projectName = route.params.project;
    const runId = route.params.id;
    const response = await fetch(
      `/api/projects/${projectName}/runs/${runId}/scalars/global_step`,
    );
    const result = await response.json();

    // New columnar format: {values: []}
    // Check if any global_step value is non-zero
    const hasNonZeroGlobalStep = result.values.some(
      (value) => value !== 0 && value !== null,
    );

    // Update all cards to use global_step if it's being used
    if (hasNonZeroGlobalStep) {
      for (const tab of tabs.value) {
        for (const card of tab.cards) {
          if (card.config.type === "line" || !card.config.type) {
            if (card.config.xMetric === "step") {
              card.config.xMetric = "global_step";
            }
          }
        }
      }
      saveLayout();
    }
  } catch (error) {
    console.error("Failed to determine default x-axis:", error);
  }
}

async function fetchMetricsForTab(forceRefresh = false) {
  try {
    console.log(
      `[fetchMetricsForTab] Starting fetch for tab: ${activeTab.value}, forceRefresh: ${forceRefresh}`,
    );
    const tab = tabs.value.find((t) => t.name === activeTab.value);
    if (!tab) {
      console.warn(`[fetchMetricsForTab] Tab not found: ${activeTab.value}`);
      return;
    }

    // Metrics that are computed on frontend (don't fetch from API)
    const computedMetrics = new Set(["walltime", "relative_walltime"]);

    const neededMetrics = new Set();
    for (const card of tab.cards) {
      // Only fetch for line plot cards
      if (card.config.type === "line" || !card.config.type) {
        if (card.config.xMetric && !computedMetrics.has(card.config.xMetric)) {
          neededMetrics.add(card.config.xMetric);
        }
        if (card.config.yMetrics) {
          for (const yMetric of card.config.yMetrics) {
            if (!computedMetrics.has(yMetric)) {
              neededMetrics.add(yMetric);
            }
          }
        }
      }
    }

    // Fetch missing or force-refresh metrics
    const projectName = route.params.project;
    const runId = route.params.id;

    for (const metric of neededMetrics) {
      // Fetch if not in cache OR if force refresh
      if (!metricDataCache.value[metric] || forceRefresh) {
        try {
          const isUpdate = !!metricDataCache.value[metric] && forceRefresh;
          console.log(
            `[fetchMetricsForTab] ${isUpdate ? "Updating" : "Fetching"} metric: ${metric}`,
          );

          // Don't URL-encode - FastAPI :path parameter handles it
          const response = await fetch(
            `/api/projects/${projectName}/runs/${runId}/scalars/${metric}`,
          );

          if (!response.ok) {
            console.warn(
              `[fetchMetricsForTab] Failed to fetch ${metric}: ${response.status}`,
            );
            // Set empty array so card can still render (just shows "no data")
            if (!metricDataCache.value[metric]) {
              metricDataCache.value[metric] = [];
            }
            continue;
          }

          const result = await response.json();

          // Check if data actually changed (for polling updates)
          if (isUpdate) {
            const oldArray = metricDataCache.value[metric];
            const oldNonNullCount = oldArray.filter((v) => v !== null).length;
            const newNonNullCount = result.values.length;

            if (oldNonNullCount === newNonNullCount) {
              console.log(
                `[fetchMetricsForTab] ${metric}: No new data (${newNonNullCount} points), skipping update`,
              );
              continue; // CRITICAL: Skip update to prevent flickering
            }

            console.log(
              `[fetchMetricsForTab] ${metric}: ${oldNonNullCount} → ${newNonNullCount} points`,
            );
          } else {
            console.log(
              `[fetchMetricsForTab] Fetched ${metric}: ${result.values?.length || 0} values`,
            );
          }

          // Validate response structure
          console.log(`[fetchMetricsForTab] ${metric} response:`, {
            hasSteps: !!result.steps,
            hasValues: !!result.values,
            stepsIsArray: Array.isArray(result.steps),
            valuesIsArray: Array.isArray(result.values),
            stepsLength: result.steps?.length,
            valuesLength: result.values?.length,
          });

          if (
            !result.steps ||
            !Array.isArray(result.steps) ||
            result.steps.length === 0
          ) {
            console.warn(
              `[fetchMetricsForTab] ${metric}: Invalid or empty steps array`,
            );
            if (!metricDataCache.value[metric]) {
              metricDataCache.value[metric] = [];
            }
            continue;
          }

          // New columnar format: {steps: [], global_steps: [], timestamps: [], values: []}
          // Convert to sparse array (indexed by step number)
          const maxStep = Math.max(...result.steps);
          console.log(
            `[fetchMetricsForTab] ${metric}: maxStep = ${maxStep}, creating array of size ${maxStep + 1}`,
          );

          // Preserve existing array size if updating to avoid range flickering
          const existingArray = metricDataCache.value[metric];
          const targetSize =
            isUpdate && existingArray
              ? Math.max(maxStep + 1, existingArray.length)
              : maxStep + 1;

          console.log(
            `[fetchMetricsForTab] ${metric}: targetSize = ${targetSize}`,
          );

          const sparseArray = new Array(targetSize).fill(null);

          for (let i = 0; i < result.steps.length; i++) {
            const step = result.steps[i];
            let value = result.values[i];

            // Convert special string markers back to numeric NaN/inf
            // Backend sends these as strings because JSON doesn't support Infinity
            if (value === "NaN") {
              value = NaN;
            } else if (value === "Infinity") {
              value = Infinity;
            } else if (value === "-Infinity") {
              value = -Infinity;
            }

            sparseArray[step] = value;
          }

          metricDataCache.value[metric] = sparseArray;
          console.log(
            `[fetchMetricsForTab] Stored ${metric} in cache: ${sparseArray.length} slots, ${sparseArray.filter((v) => v !== null).length} non-null values`,
          );

          // Also store timestamp data if available (as Unix seconds)
          if (result.timestamps && result.timestamps.length > 0) {
            const timestampArray = new Array(sparseArray.length).fill(null);
            for (let i = 0; i < result.steps.length; i++) {
              const step = result.steps[i];
              const unixSeconds = result.timestamps[i];
              if (unixSeconds !== null) {
                // Convert Unix seconds to ISO string for consistency
                timestampArray[step] = new Date(
                  unixSeconds * 1000,
                ).toISOString();
              }
            }
            metricDataCache.value[`${metric}_timestamp`] = timestampArray;
          }
        } catch (error) {
          console.error(`Failed to fetch metric ${metric}:`, error);
          // Set empty array so card can still render
          metricDataCache.value[metric] = [];
        }
      }
    }

    // Calculate walltime and relative_walltime from timestamps
    if (metricDataCache.value.timestamp) {
      const timestamps = metricDataCache.value.timestamp;
      const walltime = [];
      const relativeWalltime = [];
      let startTime = null;

      for (let i = 0; i < timestamps.length; i++) {
        if (timestamps[i]) {
          const ts = new Date(timestamps[i]).getTime() / 1000; // Convert to seconds
          walltime[i] = ts;

          if (startTime === null) {
            startTime = ts;
            relativeWalltime[i] = 0;
          } else {
            relativeWalltime[i] = ts - startTime;
          }
        } else {
          walltime[i] = null;
          relativeWalltime[i] = null;
        }
      }

      metricDataCache.value.walltime = walltime;
      metricDataCache.value.relative_walltime = relativeWalltime;
    }

    console.log(
      `[fetchMetricsForTab] Completed. Cache keys:`,
      Object.keys(metricDataCache.value),
    );
  } catch (error) {
    console.error("[fetchMetricsForTab] Error:", error);
    // Don't throw - allow page to render with partial data
  }
}

const sparseData = computed(() => {
  const data = { time: metricDataCache.value.step || [] };
  for (const [key, values] of Object.entries(metricDataCache.value)) {
    data[key] = values;
  }
  console.log(`[sparseData] Computed with ${Object.keys(data).length} metrics`);
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

// Pagination
const webglChartsPerPage = computed(() => (isMobile.value ? 2 : 12));
const defaultCardHeight = computed(() => (isMobile.value ? 280 : 400));

const paginatedCards = computed(() => {
  const allCards = currentTabCards.value;

  // Count WebGL charts (line and histogram)
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

const totalPages = computed(() => paginatedCards.value.length);

const visibleCards = computed(() => {
  if (paginatedCards.value.length === 0) return [];
  const page = Math.min(currentPage.value, paginatedCards.value.length - 1);
  return paginatedCards.value[page] || [];
});

watch(activeTab, () => {
  // Don't fetch during initialization (prevents race condition)
  if (isInitializing.value) {
    console.log("[Watch] activeTab changed during init, skipping fetch");
    return;
  }

  console.log("[Watch] activeTab changed to:", activeTab.value);
  currentPage.value = 0;
  fetchMetricsForTab();
  saveLayout();
});

// Watch route params to reinitialize when navigating between runs
watch(
  () => [route.params.project, route.params.id],
  ([newProject, newId], [oldProject, oldId]) => {
    if (oldId && (newId !== oldId || newProject !== oldProject)) {
      console.log("[Watch] Route changed - reinitializing");
      stopPolling();
      currentPage.value = 0;
      lastMetricsCount.value = 0; // Reset count for new run
      initializeExperiment();
    }
  },
);

// Polling mechanism for live updates
async function checkForUpdates() {
  if (!pollingEnabled.value) return;

  try {
    const projectName = route.params.project;
    const runId = route.params.id;

    const response = await fetch(
      `/api/projects/${projectName}/runs/${runId}/status`,
    );
    if (!response.ok) return;

    const status = await response.json();
    const newCount = status.metrics_count;

    // Check if there's new data
    if (lastMetricsCount.value > 0 && newCount > lastMetricsCount.value) {
      console.log(
        `[Poll] New data detected: ${lastMetricsCount.value} → ${newCount} metrics`,
      );

      // Re-fetch summary to check for new metrics (don't clear cache yet!)
      const summaryResponse = await fetch(
        `/api/projects/${projectName}/runs/${runId}/summary`,
      );
      if (summaryResponse.ok) {
        const summary = await summaryResponse.json();
        const oldMetrics = new Set(availableMetrics.value);
        availableMetrics.value = summary.available_data.scalars;

        // Add computed metrics
        if (availableMetrics.value.includes("timestamp")) {
          availableMetrics.value.push("walltime");
          availableMetrics.value.push("relative_walltime");
        }

        // Check for new metrics and add default cards
        const newMetrics = availableMetrics.value.filter(
          (m) => !oldMetrics.has(m),
        );
        if (newMetrics.length > 0) {
          console.log(`[Poll] New metrics found:`, newMetrics);
          await addDefaultCardsForNewMetrics(newMetrics);
        }

        // Seamlessly update existing metrics (force refetch to get new data points)
        // Old data stays visible during fetch, then smoothly updates
        await fetchMetricsForTab(true); // forceRefresh = true
      }
    }

    lastMetricsCount.value = newCount;
  } catch (error) {
    console.error("[Poll] Error checking for updates:", error);
  }
}

async function addDefaultCardsForNewMetrics(newMetrics) {
  const axisOnlyMetrics = new Set([
    "step",
    "global_step",
    "timestamp",
    "walltime",
    "relative_walltime",
  ]);

  let cardId = nextCardId.value;

  // Group by namespace
  const metricsByNamespace = new Map();
  metricsByNamespace.set("", []);

  for (const metric of newMetrics) {
    if (axisOnlyMetrics.has(metric) || removedMetrics.value.has(metric)) {
      continue;
    }

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

  // Add to main tab
  const mainTab = tabs.value.find((t) => t.name === "Metrics");
  if (mainTab) {
    for (const metric of metricsByNamespace.get("")) {
      mainTab.cards.push({
        id: `card-${cardId++}`,
        config: {
          type: "line",
          title: metric,
          widthPercent: 33,
          height: 400,
          xMetric: "step",
          yMetrics: [metric],
        },
      });
    }
  }

  // Add namespaced metrics
  for (const [namespace, metrics] of metricsByNamespace.entries()) {
    if (namespace === "") continue;

    let namespaceTab = tabs.value.find((t) => t.name === namespace);
    if (!namespaceTab) {
      namespaceTab = { name: namespace, cards: [] };
      tabs.value.push(namespaceTab);
      console.log(`[Poll] Created new tab: ${namespace}`);
    }

    for (const metric of metrics) {
      namespaceTab.cards.push({
        id: `card-${cardId++}`,
        config: {
          type: "line",
          title: metric,
          widthPercent: 33,
          height: 400,
          xMetric: "step",
          yMetrics: [metric],
        },
      });
    }
  }

  nextCardId.value = cardId;
  saveLayout();
}

function startPolling() {
  if (pollingInterval.value) return;

  console.log("[Poll] Starting polling (every 10s)");
  pollingInterval.value = setInterval(checkForUpdates, 10000);
}

function stopPolling() {
  if (pollingInterval.value) {
    console.log("[Poll] Stopping polling");
    clearInterval(pollingInterval.value);
    pollingInterval.value = null;
  }
}

function saveLayout() {
  const layout = {
    tabs: tabs.value,
    activeTab: activeTab.value,
    nextCardId: nextCardId.value,
    globalSettings: globalSettings.value,
    removedMetrics: Array.from(removedMetrics.value), // Save removed metrics list
  };
  localStorage.setItem(storageKey.value, JSON.stringify(layout));
}

function addCard() {
  showAddChartDialog.value = true;
  newChartType.value = "line";
  newChartValue.value = []; // Reset selection (array for line, will be string for others)
}

const availableChartValues = computed(() => {
  if (!availableSummary.value) return [];

  if (newChartType.value === "line") {
    // For line charts, show all metrics except axis-only ones
    const axisOnlyMetrics = new Set([
      "step",
      "global_step",
      "timestamp",
      "walltime",
      "relative_walltime",
    ]);
    return availableMetrics.value.filter((m) => !axisOnlyMetrics.has(m));
  } else if (newChartType.value === "media") {
    // Get available media from summary
    return availableSummary.value.available_data?.media || [];
  } else if (newChartType.value === "table") {
    // Get available tables from summary
    return availableSummary.value.available_data?.tables || [];
  } else if (newChartType.value === "histogram") {
    // Get available histograms from summary
    return availableSummary.value.available_data?.histograms || [];
  }
  return [];
});

function confirmAddChart() {
  const tab = tabs.value.find((t) => t.name === activeTab.value);
  if (!tab) return;

  // Validation based on chart type
  if (newChartType.value === "line") {
    if (
      !Array.isArray(newChartValue.value) ||
      newChartValue.value.length === 0
    ) {
      ElMessage.warning("Please select at least one metric");
      return;
    }
  } else {
    // Media/table/histogram require a single value
    if (!newChartValue.value) {
      ElMessage.warning(`Please select a ${newChartType.value}`);
      return;
    }
  }

  // Generate title based on selection
  let title;
  if (newChartType.value === "line") {
    // Use comma-separated list of all selected metrics
    title = newChartValue.value.join(", ");
  } else {
    title = newChartValue.value || `New ${newChartType.value}`;
  }

  const baseConfig = {
    title: title,
    widthPercent: 33,
    height: 400,
  };

  let config;
  if (newChartType.value === "line") {
    config = {
      ...baseConfig,
      type: "line",
      xMetric: "global_step",
      yMetrics: newChartValue.value, // Already an array
    };
  } else if (newChartType.value === "media") {
    config = {
      ...baseConfig,
      type: "media",
      mediaName: newChartValue.value,
      currentStep: 0,
    };
  } else if (newChartType.value === "table") {
    config = {
      ...baseConfig,
      type: "table",
      tableName: newChartValue.value,
      currentStep: 0,
    };
  } else if (newChartType.value === "histogram") {
    config = {
      ...baseConfig,
      type: "histogram",
      histogramName: newChartValue.value,
      currentStep: 0,
    };
  }

  const newCard = {
    id: `card-${nextCardId.value++}`,
    config,
  };

  tab.cards.push(newCard);
  saveLayout();
  showAddChartDialog.value = false;
  newChartValue.value = []; // Reset

  // Fetch metrics for the new chart if needed (for line charts)
  if (newChartType.value === "line") {
    fetchMetricsForTab(false); // Fetch missing metrics
  }
}

function resetLayout() {
  if (
    confirm("Reset to default layout? This will remove all customizations.")
  ) {
    localStorage.removeItem(storageKey.value);
    location.reload();
  }
}

function addTab() {
  showAddTabDialog.value = true;
  newTabName.value = "";
}

function confirmAddTab() {
  if (newTabName.value.trim()) {
    tabs.value.push({ name: newTabName.value.trim(), cards: [] });
    saveLayout();
    showAddTabDialog.value = false;
    newTabName.value = "";
  }
}

function removeTab(tabName) {
  if (tabs.value.length <= 1) {
    ElMessage.warning("Cannot remove the last tab");
    return;
  }
  tabs.value = tabs.value.filter((t) => t.name !== tabName);
  if (activeTab.value === tabName) {
    activeTab.value = tabs.value[0].name;
  }
  saveLayout();
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

  // If syncAll flag is set (shift+resize), update ALL cards in tab
  if (syncAll) {
    // During realtime drag, skip if already updating to prevent jank
    if (realtime && isUpdating.value) return;

    // Only set updating flag for final update, not realtime
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

      // Sync all rows (because width change can cause passive row wrapping)
      for (const row of rows) {
        if (row.length > 1) {
          // Determine the target height for this row
          // Priority: use height of existing cards (excluding the actively resized one if in this row)
          const isActiveCardInRow = row.includes(cardIndex);

          let targetHeight = null;

          // First, try to find height from other cards (not the actively resized one)
          const otherCards = row.filter((idx) => idx !== cardIndex);
          if (otherCards.length > 0) {
            // Use the first other card's height as the row's target height
            targetHeight = currentTab.cards[otherCards[0]].config.height;
          } else if (isActiveCardInRow) {
            // If this row only has the actively resized card, use its current height
            targetHeight = currentTab.cards[cardIndex].config.height;
          }

          if (targetHeight !== null) {
            // Sync all cards in this row to the target height
            for (const idx of row) {
              if (currentTab.cards[idx].config.height !== targetHeight) {
                console.log(
                  `[Card ${idx}] Width change caused row wrap, syncing to row height: ${targetHeight}px`,
                );
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
    // Find the card being removed to track its metrics
    const cardToRemove = tab.cards.find((c) => c.id === id);
    if (cardToRemove?.config?.yMetrics) {
      // Mark these metrics as explicitly removed
      cardToRemove.config.yMetrics.forEach((metric) => {
        removedMetrics.value.add(metric);
        console.log(`[removeCard] Marked ${metric} as removed`);
      });
    }

    tab.cards = tab.cards.filter((c) => c.id !== id);
    saveLayout();
  }
}

// Global settings functions
const hasCustomSettings = computed(() => {
  const tab = tabs.value.find((t) => t.name === activeTab.value);
  if (!tab) return false;

  // Check if any card has custom settings different from global
  return tab.cards.some((card) => {
    if (card.config.type === "line") {
      return (
        card.config.xMetric !== globalSettings.value.xAxis ||
        (card.config.smoothingMode !== undefined &&
          card.config.smoothingMode !== globalSettings.value.smoothing)
      );
    }
    if (card.config.type === "histogram") {
      return (
        card.config.histogramMode !== undefined &&
        card.config.histogramMode !== globalSettings.value.histogramMode
      );
    }
    return false;
  });
});

function applyGlobalSettings() {
  const tabIndex = tabs.value.findIndex((t) => t.name === activeTab.value);
  if (tabIndex === -1) return;

  // Create completely new tab object to force Vue reactivity
  const newCards = tabs.value[tabIndex].cards.map((card) => {
    const newConfig = { ...card.config };

    if (card.config.type === "line") {
      newConfig.xMetric = globalSettings.value.xAxis;
      newConfig.smoothingMode = globalSettings.value.smoothing;
      newConfig.smoothingValue = globalSettings.value.smoothingValue;
      newConfig.downsampleRate = globalSettings.value.downsampleRate;
    } else if (card.config.type === "histogram") {
      newConfig.histogramMode = globalSettings.value.histogramMode;
    }

    return {
      ...card,
      config: newConfig,
    };
  });

  // Replace entire tabs array to trigger reactivity
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

function onDragEnd(evt) {
  console.log("[DragEnd] Syncing heights after drag", evt);
  const currentTab = tabs.value.find((t) => t.name === activeTab.value);
  if (!currentTab) return;

  const draggedIndex = evt.newIndex;
  const oldIndex = evt.oldIndex;

  if (draggedIndex === undefined || oldIndex === undefined) {
    saveLayout();
    return;
  }

  // Calculate rows after the drag
  const newRows = calculateRows(currentTab.cards);

  // Sync ALL rows that have multiple cards
  for (const row of newRows) {
    if (row.length > 1) {
      // Find the "dominant" height: the most common height among cards in this row
      const heightCounts = {};

      for (const idx of row) {
        const h = currentTab.cards[idx].config.height;
        heightCounts[h] = (heightCounts[h] || 0) + 1;
      }

      // Find the most common height (the row's "original" height)
      let dominantHeight = null;
      let maxCount = 0;

      for (const [height, count] of Object.entries(heightCounts)) {
        if (count > maxCount) {
          maxCount = count;
          dominantHeight = parseInt(height);
        }
      }

      if (dominantHeight !== null) {
        // Sync ALL cards in this row to the dominant height
        for (const idx of row) {
          if (currentTab.cards[idx].config.height !== dominantHeight) {
            const cardLabel = idx === draggedIndex ? "(dragged)" : "(in-row)";
            console.log(
              `[DragEnd] Card ${idx} ${cardLabel} syncing to dominant row height: ${dominantHeight}px`,
            );
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
</script>

<template>
  <div class="container-main">
    <!-- Action buttons -->
    <div class="mb-6 flex items-center justify-end gap-2">
      <el-button type="primary" size="small" @click="addCard">
        <i class="i-ep-plus mr-1" />
        <span v-if="!isMobile">Add Chart</span>
        <span v-else>Add</span>
      </el-button>
      <el-button size="small" @click="addTab">
        <span v-if="!isMobile">Add Tab</span>
        <span v-else>Tab</span>
      </el-button>
      <el-button size="small" @click="isEditingTabs = !isEditingTabs">
        {{ isEditingTabs ? "Done" : "Edit" }}
      </el-button>
      <el-button size="small" @click="resetLayout" type="danger" plain>
        <i class="i-ep-refresh-left mr-1" />
        <span v-if="!isMobile">Reset Layout</span>
        <span v-else>Reset</span>
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
            <!-- Hover sync toggle button for active tab -->
            <el-button
              v-if="tab.name === activeTab && !isEditingTabs"
              size="small"
              circle
              @click.stop="toggleHoverSync"
              :title="
                hoverSyncEnabled ? 'Disable Hover Sync' : 'Enable Hover Sync'
              "
              :type="hoverSyncEnabled ? 'primary' : 'default'"
            >
              <i
                :class="
                  hoverSyncEnabled ? 'i-ep-connection' : 'i-ep-connection'
                "
              ></i>
            </el-button>
            <!-- Global settings button for active tab -->
            <el-button
              v-if="tab.name === activeTab && !isEditingTabs"
              size="small"
              circle
              @click.stop="showGlobalSettings = true"
              title="Global Tab Settings"
            >
              <i class="i-ep-setting"></i>
            </el-button>
            <!-- Remove tab button (edit mode) -->
            <el-button
              v-if="isEditingTabs && tabs.length > 1"
              icon="Close"
              size="small"
              text
              type="danger"
              @click.stop="removeTab(tab.name)"
            />
          </div>
        </template>
      </el-tab-pane>
    </el-tabs>

    <!-- Pagination controls (if needed) -->
    <div
      v-if="totalPages > 1"
      class="flex items-center justify-center gap-4 mb-4"
    >
      <el-button
        size="small"
        :disabled="currentPage === 0"
        @click="currentPage--"
        icon="ArrowLeft"
      >
        Previous
      </el-button>
      <span class="text-sm text-gray-600 dark:text-gray-400">
        Page {{ currentPage + 1 }} / {{ totalPages }}
        <span class="text-xs ml-2">
          ({{ visibleCards.length }} charts, max {{ webglChartsPerPage }} WebGL
          per page)
        </span>
      </span>
      <el-button
        size="small"
        :disabled="currentPage >= totalPages - 1"
        @click="currentPage++"
        icon="ArrowRight"
      >
        Next
      </el-button>
    </div>

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
        :experiment-id="route.params.id"
        :project="route.params.project"
        :sparse-data="sparseData"
        :available-metrics="availableMetrics"
        :initial-config="{
          ...card.config,
          height: isMobile ? defaultCardHeight : card.config.height,
        }"
        :tab-name="activeTab"
        :hover-sync-enabled="hoverSyncEnabled"
        @update:config="updateCard"
        @remove="removeCard(card.id)"
      />
    </VueDraggable>

    <el-empty
      v-if="currentTabCards.length === 0"
      description="No charts in this tab"
    >
      <el-button type="primary" @click="addCard">Add Chart</el-button>
    </el-empty>

    <!-- Add Chart Dialog -->
    <el-dialog v-model="showAddChartDialog" title="Add Chart" width="500px">
      <el-form label-width="120px">
        <el-form-item label="Chart Type">
          <el-select
            v-model="newChartType"
            class="w-full"
            @change="newChartValue = newChartType === 'line' ? [] : ''"
          >
            <el-option label="Line Chart" value="line" />
            <el-option label="Media Viewer" value="media" />
            <el-option label="Table Viewer" value="table" />
            <el-option label="Histogram" value="histogram" />
          </el-select>
        </el-form-item>

        <!-- Line Chart: Multi-select for metrics -->
        <el-form-item v-if="newChartType === 'line'" label="Select Metrics">
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

        <!-- Media: Single-select dropdown -->
        <el-form-item v-if="newChartType === 'media'" label="Select Media">
          <el-select
            v-model="newChartValue"
            class="w-full"
            placeholder="Choose a media item"
            filterable
          >
            <el-option
              v-for="media in availableChartValues"
              :key="media"
              :label="media"
              :value="media"
            />
          </el-select>
        </el-form-item>

        <!-- Table: Single-select dropdown -->
        <el-form-item v-if="newChartType === 'table'" label="Select Table">
          <el-select
            v-model="newChartValue"
            class="w-full"
            placeholder="Choose a table"
            filterable
          >
            <el-option
              v-for="table in availableChartValues"
              :key="table"
              :label="table"
              :value="table"
            />
          </el-select>
        </el-form-item>

        <!-- Histogram: Single-select dropdown -->
        <el-form-item
          v-if="newChartType === 'histogram'"
          label="Select Histogram"
        >
          <el-select
            v-model="newChartValue"
            class="w-full"
            placeholder="Choose a histogram"
            filterable
          >
            <el-option
              v-for="hist in availableChartValues"
              :key="hist"
              :label="hist"
              :value="hist"
            />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddChartDialog = false">Cancel</el-button>
        <el-button type="primary" @click="confirmAddChart">Add</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="showAddTabDialog" title="Add Tab" width="400px">
      <el-form @submit.prevent="confirmAddTab">
        <el-form-item label="Tab Name">
          <el-input
            v-model="newTabName"
            placeholder="Enter tab name"
            @keyup.enter="confirmAddTab"
            autofocus
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddTabDialog = false">Cancel</el-button>
        <el-button type="primary" @click="confirmAddTab">Add</el-button>
      </template>
    </el-dialog>

    <!-- Global Tab Settings Dialog -->
    <el-dialog
      v-model="showGlobalSettings"
      title="Global Tab Settings"
      width="600px"
    >
      <el-alert
        v-if="hasCustomSettings"
        type="warning"
        :closable="false"
        class="mb-4"
      >
        <template #title> Some cards have custom settings </template>
        Click "Apply to All Cards" below to reset all cards to these global
        settings
      </el-alert>

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
          <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
            -1 = Adaptive (recommended), 1 = No downsampling
          </div>
        </el-form-item>

        <el-divider content-position="left">Histogram Settings</el-divider>

        <el-form-item label="Histogram Mode">
          <el-select v-model="globalSettings.histogramMode" class="w-full">
            <el-option label="Single Step" value="single" />
            <el-option label="Distribution Flow" value="flow" />
          </el-select>
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
