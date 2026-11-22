import { fileURLToPath, URL } from "node:url"
import { defineConfig } from "vite"
import vue from "@vitejs/plugin-vue"
import UnoCSS from "unocss/vite"

export default defineConfig({
  plugins: [vue(), UnoCSS()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:48889",
        changeOrigin: true,
      },
      "/healthz": {
        target: "http://127.0.0.1:48889",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "../../packages/kohakuboard-web/src/kohakuboard_web/static",
    emptyOutDir: true,
    sourcemap: false,
  },
})
