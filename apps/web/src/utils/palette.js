/**
 * KohakuTerrarium 5-gem palette & deterministic run color assignment.
 */

export const GEM_COLORS = {
  iolite: "#5A4FCF",
  sapphire: "#0F52BA",
  aquamarine: "#4C9989",
  taaffeite: "#A57EAE",
  amber: "#D4920A",
  sage: "#5A9E6F",
  coral: "#D46B6B",
}

// High-contrast 10-run gemstone-harmonized palette readable on both light stone & dark obsidian.
export const RUN_PALETTE = [
  "#5A4FCF", // iolite
  "#4C9989", // aquamarine
  "#D4920A", // amber
  "#0F52BA", // sapphire
  "#A57EAE", // taaffeite
  "#D46B6B", // coral
  "#5A9E6F", // sage
  "#C96F3B", // warm copper
  "#2F80A6", // deep teal
  "#8C52A8", // amethyst
]

function fnv1a(str) {
  let hash = 2166136261
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

export function colorForRun(runId, index = -1) {
  if (index >= 0 && index < RUN_PALETTE.length) {
    return RUN_PALETTE[index]
  }
  const h = fnv1a(String(runId))
  if (index < 0) {
    return RUN_PALETTE[h % RUN_PALETTE.length]
  }
  const hue = (h * 137.508) % 360
  return `hsl(${Math.round(hue)} 56% 52%)`
}

export function statusColor(state) {
  switch (state) {
    case "running":
      return GEM_COLORS.aquamarine
    case "finished":
      return GEM_COLORS.sage
    case "crashed":
      return GEM_COLORS.coral
    case "interrupted":
      return GEM_COLORS.amber
    default:
      return "#8A8480"
  }
}

export function chartChrome(isDark) {
  return isDark
    ? {
        axis: "#a39a91",
        grid: "rgba(255, 255, 255, 0.06)",
        ticks: "rgba(255, 255, 255, 0.10)",
      }
    : {
        axis: "#6c655d",
        grid: "rgba(40, 34, 28, 0.07)",
        ticks: "rgba(40, 34, 28, 0.12)",
      }
}
