import {
  defineConfig,
  presetWind3,
  presetIcons,
  transformerDirectives,
} from "unocss"

export default defineConfig({
  presets: [
    presetWind3(),
    presetIcons({
      scale: 1.15,
      extraProperties: {
        display: "inline-block",
        "vertical-align": "middle",
      },
    }),
  ],
  transformers: [transformerDirectives()],
  preflights: [
    {
      getCSS: ({ theme }) => {
        const declarations = []
        for (const [name, value] of Object.entries(theme.colors || {})) {
          if (typeof value === "string") {
            declarations.push(`--kt-color-${name}: ${value};`)
            continue
          }
          for (const [shade, hex] of Object.entries(value)) {
            const suffix = shade === "DEFAULT" ? "" : `-${shade}`
            declarations.push(`--kt-color-${name}${suffix}: ${hex};`)
          }
        }
        return `:root {\n  ${declarations.join("\n  ")}\n}`
      },
    },
  ],
  theme: {
    colors: {
      // KohakuTerrarium 5 Gem Accent Colors
      sapphire: { light: "#D6E3F8", DEFAULT: "#0F52BA", shadow: "#082567" },
      aquamarine: { light: "#D4EDE8", DEFAULT: "#4C9989", shadow: "#1B6B5A" },
      taaffeite: { light: "#E8D5ED", DEFAULT: "#A57EAE", shadow: "#6B4670" },
      iolite: { light: "#DDD0F0", DEFAULT: "#5A4FCF", shadow: "#312A7A" },
      amber: { light: "#F5E6C8", DEFAULT: "#D4920A", shadow: "#8B5E00" },
      // Functional status accents
      coral: { light: "#F5D5D5", DEFAULT: "#D46B6B", shadow: "#8B3A3A" },
      sage: { light: "#D5E8DA", DEFAULT: "#5A9E6F", shadow: "#3A6B48" },
      // Warm stone / obsidian ramp
      warm: {
        50: "#F7F5F2",
        100: "#EFECE7",
        200: "#E0DBD4",
        300: "#C5BFB7",
        400: "#A09A92",
        500: "#8A8480",
        600: "#6A645F",
        700: "#4A4540",
        800: "#3A3632",
        900: "#2A2724",
        950: "#1A1816",
      },
    },
  },
  shortcuts: {
    num: "tabular-nums font-mono",
    "btn-primary":
      "px-3 py-1.5 rounded-lg bg-iolite text-white hover:bg-iolite-shadow transition-colors font-medium text-xs cursor-pointer",
    "btn-secondary":
      "px-2.5 py-1.5 rounded-lg bg-warm-100 dark:bg-warm-800 text-warm-700 dark:text-warm-200 hover:bg-warm-200 dark:hover:bg-warm-700 transition-colors font-medium text-xs border border-warm-200/70 dark:border-warm-700/70 cursor-pointer",
    "btn-ghost":
      "px-2 py-1 rounded-md text-xs text-warm-600 dark:text-warm-300 hover:bg-warm-200/60 dark:hover:bg-warm-700/50 transition-colors cursor-pointer disabled:opacity-40",
  },
})
