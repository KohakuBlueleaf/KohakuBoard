import { createApp } from "vue";
import { createPinia } from "pinia";
import { createRouter, createWebHistory } from "vue-router";
import { routes } from "vue-router/auto-routes";
import App from "./App.vue";

// Import UnoCSS
import "virtual:uno.css";
import "@unocss/reset/tailwind.css";

// Import Element Plus
import "element-plus/dist/index.css";
import "element-plus/theme-chalk/dark/css-vars.css";

// Import custom highlight.js theme for syntax highlighting
import "./styles/highlight-theme.css";

const app = createApp(App);
const pinia = createPinia();

const router = createRouter({
  history: createWebHistory(),
  routes,
});

app.use(pinia);
app.use(router);
app.mount("#app");
