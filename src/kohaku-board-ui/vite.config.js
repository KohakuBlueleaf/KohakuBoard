import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import VueRouter from 'unplugin-vue-router/vite'
import UnoCSS from 'unocss/vite'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import { fileURLToPath, URL } from 'node:url'
import { cpSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'

// Plugin to copy build output to multiple directories
function copyToMultipleOutputs(additionalOutputs) {
  return {
    name: 'copy-to-multiple-outputs',
    closeBundle() {
      const primaryOut = resolve(__dirname, 'dist')
      for (const dest of additionalOutputs) {
        const destPath = resolve(__dirname, dest)
        try {
          rmSync(destPath, { recursive: true, force: true })
          cpSync(primaryOut, destPath, { recursive: true })
          console.log(`✓ Copied build to ${dest}`)
        } catch (err) {
          console.error(`✗ Failed to copy to ${dest}:`, err.message)
        }
      }
    }
  }
}

export default defineConfig({
  plugins: [
    VueRouter({
      routesFolder: 'src/pages',
      dts: 'src/typed-router.d.ts',
      extensions: ['.vue'],
      exclude: ['**/components/**']
    }),
    vue(),
    UnoCSS(),
    AutoImport({
      imports: ['vue', 'pinia', 'vue-router', { 'vue-router/auto': ['useRoute', 'useRouter'] }],
      resolvers: [ElementPlusResolver()],
      dts: 'src/auto-imports.d.ts'
    }),
    Components({
      resolvers: [ElementPlusResolver()],
      dts: 'src/components.d.ts',
      dirs: ['src/components']
    }),
    copyToMultipleOutputs(['../kohakuboard/static'])
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  },
  server: {
    port: 5175,
    proxy: {
      '/api': {
        target: 'http://localhost:48889',
        changeOrigin: true
      }
    }
  }
})
