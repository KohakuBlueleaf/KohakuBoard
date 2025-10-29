<script setup>
const sections = [
  {
    title: "Getting Started",
    description: "Quick start guide, installation, and first experiment",
    path: "/docs/getting-started",
    icon: "i-ep-Document",
    color: "text-blue-600 dark:text-blue-400",
  },
  {
    title: "Architecture",
    description: "Deep dive into logging mechanism, storage, and data flow",
    path: "/docs/architecture",
    icon: "i-ep-Connection",
    color: "text-orange-600 dark:text-orange-400",
  },
  {
    title: "Python API Reference",
    description: "Complete API documentation for the Python client",
    path: "/docs/api",
    icon: "i-ep-Document",
    color: "text-green-600 dark:text-green-400",
  },
  {
    title: "CLI Tools",
    description: "Command-line interface for managing boards",
    path: "/docs/cli",
    icon: "i-ep-Document",
    color: "text-purple-600 dark:text-purple-400",
  },
];
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 py-12 px-4">
    <div class="max-w-6xl mx-auto">
      <!-- Header -->
      <div class="mb-12">
        <h1 class="text-4xl font-bold mb-4 text-gray-900 dark:text-gray-100">
          KohakuBoard Documentation
        </h1>
        <p class="text-xl text-gray-600 dark:text-gray-400">
          High-performance experiment tracking for ML workflows
        </p>
      </div>

      <!-- Quick Links -->
      <div
        class="mb-12 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6"
      >
        <h2 class="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
          🚀 Quick Start
        </h2>
        <div class="space-y-3 text-sm">
          <div>
            <p class="font-medium text-gray-900 dark:text-gray-100 mb-1">
              Installation:
            </p>
            <pre
              class="bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 overflow-x-auto"
            >
pip install -e src/kohakuboard/</pre
            >
          </div>
          <div>
            <p class="font-medium text-gray-900 dark:text-gray-100 mb-1">
              Your first experiment:
            </p>
            <pre
              class="bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 overflow-x-auto"
            >
from kohakuboard.client import Board

board = Board(name="my-experiment")
board.log(loss=0.5, accuracy=0.95)</pre
            >
          </div>
          <div>
            <p class="font-medium text-gray-900 dark:text-gray-100 mb-1">
              View results:
            </p>
            <pre
              class="bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 overflow-x-auto"
            >
kobo open ./kohakuboard --browser</pre
            >
          </div>
        </div>
      </div>

      <!-- Documentation Sections -->
      <h2 class="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">
        📚 Documentation Sections
      </h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
        <router-link
          v-for="section in sections"
          :key="section.path"
          :to="section.path"
          class="block bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg hover:border-blue-400 dark:hover:border-blue-600 transition-all"
        >
          <div class="flex items-center gap-3 mb-3">
            <div :class="[section.icon, section.color]" class="text-3xl" />
            <h3 class="text-xl font-bold text-gray-900 dark:text-gray-100">
              {{ section.title }}
            </h3>
          </div>
          <p class="text-sm text-gray-600 dark:text-gray-400">
            {{ section.description }}
          </p>
        </router-link>
      </div>

      <!-- Key Features -->
      <div class="mb-12">
        <h2 class="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">
          ✨ Key Features
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div
            class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 class="font-semibold mb-2 text-gray-900 dark:text-gray-100">
              🚄 Non-Blocking Performance
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">
              Background writer process handles all I/O. Zero training overhead.
            </p>
          </div>

          <div
            class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 class="font-semibold mb-2 text-gray-900 dark:text-gray-100">
              📊 Rich Data Types
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">
              Scalars, images, videos, tables, histograms - all in one unified
              API.
            </p>
          </div>

          <div
            class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 class="font-semibold mb-2 text-gray-900 dark:text-gray-100">
              🎯 No Step Inflation
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">
              Log multiple values at the same step. Solves the histogram step
              problem.
            </p>
          </div>

          <div
            class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 class="font-semibold mb-2 text-gray-900 dark:text-gray-100">
              💾 Flexible Storage
            </h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">
              3 backend options: Hybrid (default), DuckDB, Parquet. Choose what
              fits.
            </p>
          </div>
        </div>
      </div>

      <!-- Resources -->
      <div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
        <h2 class="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">
          📖 Additional Resources
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="https://github.com/KohakuBlueleaf/KohakuHub/tree/main/examples"
            target="_blank"
            rel="noopener noreferrer"
            class="flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
          >
            <span class="i-ep-FolderOpened" />
            Examples
          </a>
          <a
            href="https://github.com/KohakuBlueleaf/KohakuHub"
            target="_blank"
            rel="noopener noreferrer"
            class="flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
          >
            <span class="i-ep-Link" />
            GitHub
          </a>
          <a
            href="https://discord.gg/xWYrkyvJ2s"
            target="_blank"
            rel="noopener noreferrer"
            class="flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline"
          >
            <span class="i-ep-ChatLineSquare" />
            Discord
          </a>
        </div>
      </div>
    </div>
  </div>
</template>
