import { defineConfig, presetUno, presetIcons } from 'unocss'

export default defineConfig({
  presets: [
    presetUno(),
    presetIcons()
  ],
  theme: {
    colors: {
      primary: {
        DEFAULT: '#3b82f6',
        dark: '#2563eb'
      }
    }
  },
  shortcuts: {
    'card': 'bg-white dark:bg-gray-900 rounded-lg shadow-md p-4 border border-gray-200 dark:border-gray-800',
    'card-title': 'text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4',
    'btn': 'px-4 py-2 rounded-md font-medium transition-colors shadow-sm',
    'btn-primary': 'btn bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white',
    'input': 'px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent outline-none'
  }
})
