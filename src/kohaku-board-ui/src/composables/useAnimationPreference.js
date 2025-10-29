import { ref, watch, onMounted } from "vue";

// Global state for animation preference
const animationsEnabled = ref(true);
let initialized = false;

/**
 * Composable for managing animation preferences
 * - Respects prefers-reduced-motion media query
 * - Provides user toggle override
 * - Persists preference to localStorage
 * - Applies global CSS class to disable animations
 */
export function useAnimationPreference() {
  // Initialize only once globally
  if (!initialized) {
    initialized = true;

    onMounted(() => {
      // Check localStorage first
      const saved = localStorage.getItem("animations-enabled");

      if (saved !== null) {
        // User has set a preference
        animationsEnabled.value = saved === "true";
      } else {
        // Check system preference
        const prefersReducedMotion = window.matchMedia(
          "(prefers-reduced-motion: reduce)",
        ).matches;
        animationsEnabled.value = !prefersReducedMotion;
      }

      updateGlobalClass();

      // Listen for system preference changes
      const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
      const handleChange = (e) => {
        // Only auto-update if user hasn't set explicit preference
        if (localStorage.getItem("animations-enabled") === null) {
          animationsEnabled.value = !e.matches;
          updateGlobalClass();
        }
      };

      // Modern browsers
      if (mediaQuery.addEventListener) {
        mediaQuery.addEventListener("change", handleChange);
      } else {
        // Fallback for older browsers
        mediaQuery.addListener(handleChange);
      }
    });

    // Watch for changes and update global class
    watch(animationsEnabled, () => {
      updateGlobalClass();
      localStorage.setItem(
        "animations-enabled",
        animationsEnabled.value.toString(),
      );
    });
  }

  function updateGlobalClass() {
    if (animationsEnabled.value) {
      document.documentElement.classList.remove("no-animations");
    } else {
      document.documentElement.classList.add("no-animations");
    }
  }

  function toggleAnimations() {
    animationsEnabled.value = !animationsEnabled.value;
  }

  return {
    animationsEnabled,
    toggleAnimations,
  };
}
