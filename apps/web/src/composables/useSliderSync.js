import { onBeforeUnmount, onMounted, ref, watch } from "vue"

/**
 * Global Shift+drag step synchronization across all Media, Distribution, and Table cards.
 * Ported from KohakuBoard v1's useSliderSync composable.
 */
export const isShiftPressed = ref(false)
const sliderRegistry = new Map()

let listenerInstalled = false

export function initializeSliderSync() {
  if (listenerInstalled || typeof window === "undefined") return
  listenerInstalled = true
  window.addEventListener("keydown", (e) => {
    if (e.key === "Shift") isShiftPressed.value = true
  })
  window.addEventListener("keyup", (e) => {
    if (e.key === "Shift") isShiftPressed.value = false
  })
  window.addEventListener("blur", () => {
    isShiftPressed.value = false
  })
}

function findClosestStep(steps, targetStep) {
  if (!steps || steps.length === 0) return null
  let best = steps[0]
  let bestDiff = Math.abs(steps[0] - targetStep)
  for (let i = 1; i < steps.length; i++) {
    const d = Math.abs(steps[i] - targetStep)
    if (d < bestDiff) {
      bestDiff = d
      best = steps[i]
    }
  }
  return best
}

/**
 * Register a card's step slider for Shift+drag synchronization.
 *
 * @param {import('vue').Ref<string> | string} idRef
 * @param {import('vue').Ref<number[]>} stepsRef - Sorted array of available numeric steps
 * @param {(step: number) => void} onStepChange
 */
export function useSliderSync(idRef, stepsRef, onStepChange) {
  initializeSliderSync()

  const getId = () => (typeof idRef === "string" ? idRef : idRef.value)

  function register() {
    const id = getId()
    if (!id) return
    sliderRegistry.set(id, {
      steps: stepsRef.value || [],
      onStepChange,
    })
  }

  function unregister() {
    const id = getId()
    if (id) sliderRegistry.delete(id)
  }

  function triggerSync(currentStep) {
    if (!isShiftPressed.value || !Number.isFinite(currentStep)) return
    const selfId = getId()
    for (const [otherId, entry] of sliderRegistry.entries()) {
      if (otherId === selfId) continue
      const closest = findClosestStep(entry.steps, currentStep)
      if (closest !== null) entry.onStepChange(closest)
    }
  }

  onMounted(register)
  onBeforeUnmount(unregister)
  watch(stepsRef, register, { deep: true })

  return {
    isShiftPressed,
    triggerSync,
  }
}
