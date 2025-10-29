<template>
  <footer
    class="bg-gray-100 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-8 transition-colors"
  >
    <div class="max-w-full mx-auto px-6 py-8">
      <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8">
        <!-- About -->
        <div>
          <h3 class="font-semibold mb-3 text-gray-900 dark:text-gray-100">
            KohakuBoard
          </h3>
          <p class="text-sm text-gray-600 dark:text-gray-400">
            High-performance ML experiment tracking & visualization
          </p>
        </div>

        <!-- Documentation -->
        <div>
          <h3 class="font-semibold mb-3 text-gray-900 dark:text-gray-100">
            Documentation
          </h3>
          <div class="flex flex-col gap-2 text-sm">
            <router-link
              to="/docs"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              Overview
            </router-link>
            <router-link
              to="/docs/getting-started"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              Getting Started
            </router-link>
            <router-link
              to="/docs/architecture"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              Architecture
            </router-link>
            <router-link
              to="/docs/api"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              API Reference
            </router-link>
            <router-link
              to="/docs/cli"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              CLI Tools
            </router-link>
          </div>
        </div>

        <!-- Resources -->
        <div>
          <h3 class="font-semibold mb-3 text-gray-900 dark:text-gray-100">
            Resources
          </h3>
          <div class="flex flex-col gap-2 text-sm">
            <a
              href="https://github.com/KohakuBlueleaf/KohakuHub"
              target="_blank"
              rel="noopener noreferrer"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              GitHub
            </a>
            <a
              href="https://github.com/KohakuBlueleaf/KohakuHub/tree/main/examples"
              target="_blank"
              rel="noopener noreferrer"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              Examples
            </a>
            <a
              href="https://github.com/KohakuBlueleaf/KohakuHub/issues"
              target="_blank"
              rel="noopener noreferrer"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              Issues
            </a>
          </div>
        </div>

        <!-- Community -->
        <div>
          <h3 class="font-semibold mb-3 text-gray-900 dark:text-gray-100">
            Community
          </h3>
          <div class="flex flex-col gap-2 text-sm">
            <a
              href="https://discord.gg/xWYrkyvJ2s"
              target="_blank"
              rel="noopener noreferrer"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              Discord
            </a>
            <a
              href="https://github.com/KohakuBlueleaf/KohakuHub/discussions"
              target="_blank"
              rel="noopener noreferrer"
              class="text-gray-600 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
            >
              Discussions
            </a>
          </div>
        </div>
      </div>

      <!-- Copyright -->
      <div
        class="mt-8 pt-8 border-t border-gray-300 dark:border-gray-600 text-center text-sm text-gray-600 dark:text-gray-400"
      >
        <p>© 2025 KohakuBoard. Part of KohakuHub Project</p>
        <p class="mt-2">
          Licensed under
          <a
            href="https://kblueleaf.net/documents/kohaku-code-license/?%5BYour%20Organization/Name%5D=KohakuBlueLeaf&%5BYear%5D=2025&[Your%20Contact%20Email]=kohaku@kblueleaf.net&[Your%20Jurisdiction]=Taiwan"
            target="_blank"
            rel="noopener noreferrer"
            class="text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Kohaku Software License 1.0
          </a>
        </p>
      </div>
    </div>
  </footer>
</template>
