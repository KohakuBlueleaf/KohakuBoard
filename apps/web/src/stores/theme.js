import { defineStore } from "pinia"
import { ref, watch } from "vue"

const STORAGE_KEY = "kobo.theme.v2"

export const useThemeStore = defineStore("theme", () => {
  const saved = typeof localStorage !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null
  const isDark = ref(saved === "dark")

  function apply() {
    if (typeof document === "undefined") return
    document.documentElement.classList.toggle("dark", isDark.value)
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(STORAGE_KEY, isDark.value ? "dark" : "light")
    }
  }

  function toggle() {
    isDark.value = !isDark.value
  }

  function setDark(val) {
    isDark.value = Boolean(val)
  }

  watch(isDark, apply, { immediate: true })

  return { isDark, toggle, setDark }
})
