import { ref, onMounted, onUnmounted } from "vue";

// Global state for slider synchronization
const isShiftPressed = ref(false);
const sliderSyncCallbacks = new Map();

// Track shift key state globally
function setupShiftKeyListener() {
  const handleKeyDown = (e) => {
    if (e.key === "Shift") {
      isShiftPressed.value = true;
    }
  };

  const handleKeyUp = (e) => {
    if (e.key === "Shift") {
      isShiftPressed.value = false;
    }
  };

  window.addEventListener("keydown", handleKeyDown);
  window.addEventListener("keyup", handleKeyUp);

  return () => {
    window.removeEventListener("keydown", handleKeyDown);
    window.removeEventListener("keyup", handleKeyUp);
  };
}

// Initialize global shift key listener (called once from App.vue or main component)
let cleanupShiftListener = null;
export function initializeSliderSync() {
  if (!cleanupShiftListener) {
    cleanupShiftListener = setupShiftKeyListener();
  }
}

/**
 * Register a slider for synchronized sliding
 * @param {string} id - Unique identifier for this slider
 * @param {Array} data - Array of data items with step property
 * @param {Function} onStepChange - Callback to update step: (newStep: number) => void
 */
export function useSliderSync(id, data, onStepChange) {
  const register = () => {
    sliderSyncCallbacks.set(id, {
      data: data.value || data,
      onStepChange,
    });
  };

  const unregister = () => {
    sliderSyncCallbacks.delete(id);
  };

  /**
   * Trigger synchronization across all sliders
   * @param {number} currentStep - The step value from the current slider
   */
  const triggerSync = (currentStep) => {
    if (!isShiftPressed.value) return;

    // Sync all other sliders to the closest step
    for (const [
      otherId,
      { data: otherData, onStepChange: otherCallback },
    ] of sliderSyncCallbacks.entries()) {
      if (otherId === id) continue; // Skip self

      // Find closest step in the other slider's data
      const closestEntry = findClosestStep(otherData, currentStep);
      if (closestEntry) {
        otherCallback(closestEntry.step);
      }
    }
  };

  /**
   * Find the closest step in a dataset to the target step
   * @param {Array} data - Array of items with step property
   * @param {number} targetStep - The step to find closest to
   * @returns {Object|null} - The closest data entry or null
   */
  function findClosestStep(data, targetStep) {
    if (!data || data.length === 0) return null;

    let closest = data[0];
    let minDiff = Math.abs(data[0].step - targetStep);

    for (const entry of data) {
      const diff = Math.abs(entry.step - targetStep);
      if (diff < minDiff) {
        minDiff = diff;
        closest = entry;
      }
    }

    return closest;
  }

  onMounted(() => {
    register();
  });

  onUnmounted(() => {
    unregister();
  });

  // Re-register when data changes
  watch(
    () => data.value || data,
    () => {
      register();
    },
    { deep: true },
  );

  return {
    isShiftPressed,
    triggerSync,
  };
}
