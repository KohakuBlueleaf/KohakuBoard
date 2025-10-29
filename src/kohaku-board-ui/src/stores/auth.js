// src/kohaku-board-ui/src/stores/auth.js
import { defineStore } from "pinia";
import { authAPI } from "@/utils/api";

export const useAuthStore = defineStore("auth", {
  state: () => ({
    user: null,
    userOrganizations: [],
    loading: false,
    initialized: false,
  }),

  getters: {
    isAuthenticated: (state) => !!state.user,
    username: (state) => state.user?.username || null,
    organizations: (state) => state.userOrganizations,
    organizationNames: (state) =>
      state.userOrganizations.map((org) => org.name),
  },

  actions: {
    /**
     * Login user
     * @param {Object} credentials - {username, password}
     */
    async login(credentials) {
      this.loading = true;
      try {
        const { data } = await authAPI.login(credentials);
        await this.fetchUserInfo();
        return data;
      } finally {
        this.loading = false;
      }
    },

    /**
     * Register new user
     * @param {Object} userData - {username, email, password}
     */
    async register(userData) {
      this.loading = true;
      try {
        const { data } = await authAPI.register(userData);
        return data;
      } finally {
        this.loading = false;
      }
    },

    /**
     * Logout current user
     */
    async logout() {
      try {
        await authAPI.logout();
      } finally {
        this.user = null;
        this.userOrganizations = [];
      }
    },

    /**
     * Fetch current user info with organizations
     */
    async fetchUserInfo() {
      try {
        const { data } = await authAPI.me();
        this.user = data;

        // Fetch user's organizations
        if (data.username) {
          const orgsData = await authAPI.listUserOrgs(data.username);
          this.userOrganizations = orgsData.data.organizations || [];
        }

        return data;
      } catch (err) {
        this.user = null;
        this.userOrganizations = [];
        throw err;
      }
    },

    /**
     * Initialize auth state (restore session on app load)
     */
    async init() {
      if (this.initialized) return;

      this.initialized = true;

      try {
        await this.fetchUserInfo();
      } catch (err) {
        // Session expired or invalid
        this.user = null;
        this.userOrganizations = [];
      }
    },

    /**
     * Check if user has write access to a namespace
     * @param {string} namespace - Namespace to check
     * @returns {boolean}
     */
    canWriteToNamespace(namespace) {
      if (!this.isAuthenticated) return false;
      return (
        this.username === namespace ||
        this.organizationNames.includes(namespace)
      );
    },
  },
});
