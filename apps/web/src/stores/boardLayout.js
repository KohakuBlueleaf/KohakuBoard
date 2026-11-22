import { defineStore } from "pinia"
import { ref } from "vue"

const STORAGE_PREFIX = "kobo.board.v3."
const TEMPLATE_PREFIX = "kobo.board.tpl.v3."

export const WIDTH_SNAPS = [
  { p: 16.666, l: "1/6" },
  { p: 20.0, l: "1/5" },
  { p: 25.0, l: "1/4" },
  { p: 33.333, l: "1/3" },
  { p: 50.0, l: "1/2" },
  { p: 66.666, l: "2/3" },
  { p: 100.0, l: "Full" },
]

export function widthSnapLabel(pct) {
  const found = WIDTH_SNAPS.find((s) => Math.abs(s.p - pct) < 0.6)
  return found ? found.l : `${Math.round(pct)}%`
}

export function nearestWidthSnap(ratioPercent) {
  let closest = WIDTH_SNAPS[0]
  let minDiff = Math.abs(ratioPercent - closest.p)
  for (const s of WIDTH_SNAPS) {
    const d = Math.abs(ratioPercent - s.p)
    if (d < minDiff) {
      minDiff = d
      closest = s
    }
  }
  return closest
}

function uid(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`
}

export function createDefaultTabSettings() {
  return {
    xAxis: "global_step",
    smoothingMode: "ema", // "none" | "ema" | "ma" | "gaussian"
    smoothingValue: 0,
    logY: false,
    showOriginal: true,
    showMarkers: false,
    histogramMode: "single", // "single" | "flow"
  }
}

function createSection(title, autoSubNamespace = null) {
  return {
    id: autoSubNamespace !== null ? `sec:${autoSubNamespace || "_root"}` : uid("sec"),
    title,
    autoSubNamespace,
    collapsed: false,
    cards: [],
  }
}

function createTab(name, autoNamespace = null) {
  return {
    id: autoNamespace !== null ? `ns:${autoNamespace || "_root"}` : uid("tab"),
    name,
    autoNamespace,
    settings: createDefaultTabSettings(),
    sections: [],
  }
}

function defaultBoardState() {
  return {
    schemaVersion: 3,
    activeTabId: null,
    defaultWidth: 33.333,
    defaultHeight: 260,
    tabs: [],
    removedKeys: [],
  }
}

/**
 * Split a key like "train/loss/ce" into { tabNs: "train", secNs: "loss" }.
 * - "loss" -> { tabNs: "", secNs: "" } (tab "Metrics", section "General")
 * - "train/loss" -> { tabNs: "train", secNs: "" } (tab "train", section "General")
 * - "train/loss/ce" -> { tabNs: "train", secNs: "loss" } (tab "train", section "loss")
 */
export function splitNamespace(name) {
  const parts = String(name || "").split("/").filter(Boolean)
  if (parts.length <= 1) return { tabNs: "", secNs: "" }
  if (parts.length === 2) return { tabNs: parts[0], secNs: "" }
  return { tabNs: parts[0], secNs: parts[1] }
}

/**
 * Compute visual flex-wrap rows of cards for OG v1 row-height alignment.
 */
function calculateRows(cards) {
  const rows = []
  let current = []
  let sum = 0
  for (let i = 0; i < cards.length; i++) {
    const w = cards[i].widthPercent || 50
    if (sum > 0 && sum + w > 102) {
      rows.push(current)
      current = []
      sum = 0
    }
    current.push(i)
    sum += w
  }
  if (current.length > 0) rows.push(current)
  return rows
}

function autoKeyForCard(card) {
  if (!card || card.custom) return null
  if (card.kind === "scalar" && card.metrics?.length === 1) return `scalar:${card.metrics[0]}`
  if (card.kind === "media" && card.mediaName) return `media:${card.mediaName}`
  if (card.kind === "distribution" && card.distName) {
    return `${card.distKind || "histogram"}:${card.distName}`
  }
  if (card.kind === "table" && card.tableName) return `table:${card.tableName}`
  return null
}

/**
 * Unified Multimodal Board Layout Store for BOTH Project View and Run View.
 *
 * Hierarchy:
 *   Scope (proj:<p> | run:<p>:<id>)
 *     └─ tabs[] (auto-created by namespace across ALL modalities + user custom tabs)
 *          └─ sections[] (W&B-style collapsible sections, auto-grouped by sub-namespace + user custom sections)
 *               └─ cards[] (scalar | media | distribution | table — independently resizable & draggable)
 */
export const useBoardLayoutStore = defineStore("boardLayout", () => {
  const byScope = ref({})
  // Track in-flight card drag so tabs, sections, and cards can highlight drop targets smoothly.
  const draggingCard = ref(null) // { scopeKey, tabId, sectionId, cardId }

  function _load(scopeKey, fallbackTemplateKey = null) {
    if (typeof localStorage === "undefined") return defaultBoardState()
    try {
      const raw =
        localStorage.getItem(STORAGE_PREFIX + scopeKey) ||
        (fallbackTemplateKey ? localStorage.getItem(TEMPLATE_PREFIX + fallbackTemplateKey) : null)
      if (!raw) return defaultBoardState()
      const parsed = JSON.parse(raw)
      if (!parsed || parsed.schemaVersion !== 3 || !Array.isArray(parsed.tabs)) {
        return defaultBoardState()
      }
      // Ensure every tab has settings & sections array
      for (const t of parsed.tabs) {
        t.settings = { ...createDefaultTabSettings(), ...(t.settings || {}) }
        if (!Array.isArray(t.sections)) t.sections = []
      }
      return { ...defaultBoardState(), ...parsed }
    } catch {
      return defaultBoardState()
    }
  }

  function _save(scopeKey) {
    if (typeof localStorage === "undefined") return
    try {
      localStorage.setItem(STORAGE_PREFIX + scopeKey, JSON.stringify(byScope.value[scopeKey]))
    } catch {
      // ignore storage quota error
    }
  }

  function ensure(scopeKey, fallbackTemplateKey = null) {
    if (!byScope.value[scopeKey]) {
      byScope.value[scopeKey] = _load(scopeKey, fallbackTemplateKey)
    }
    return byScope.value[scopeKey]
  }

  function _findTab(st, tabId) {
    return st.tabs.find((t) => t.id === tabId) || null
  }

  function _findSection(tab, sectionId) {
    if (!tab) return null
    return tab.sections.find((s) => s.id === sectionId) || null
  }

  function _ensureAutoTab(st, tabNs) {
    const id = `ns:${tabNs || "_root"}`
    let tab = st.tabs.find((t) => t.id === id || t.autoNamespace === tabNs)
    if (!tab) {
      const label = tabNs === "" ? "Metrics" : tabNs
      tab = createTab(label, tabNs)
      // Keep root "Metrics" tab first if added later
      if (tabNs === "") st.tabs.unshift(tab)
      else st.tabs.push(tab)
    }
    return tab
  }

  function _ensureAutoSection(tab, secNs) {
    const id = `sec:${secNs || "_root"}`
    let sec = tab.sections.find((s) => s.id === id || s.autoSubNamespace === secNs)
    if (!sec) {
      const title = secNs === "" ? "General" : secNs
      sec = createSection(title, secNs)
      if (secNs === "") tab.sections.unshift(sec)
      else tab.sections.push(sec)
    }
    return sec
  }

  /**
   * Synchronize all logged modalities (scalars, media, histograms, kdes, tables)
   * into auto-created namespace Tabs & collapsible sub-namespace Sections.
   */
  function syncCatalog(scopeKey, catalog, fallbackTemplateKey = null) {
    const st = ensure(scopeKey, fallbackTemplateKey)
    const removed = new Set(st.removedKeys || [])
    const existingAuto = new Set()
    for (const t of st.tabs) {
      for (const s of t.sections) {
        for (const c of s.cards) {
          const k = autoKeyForCard(c)
          if (k) existingAuto.add(k)
        }
      }
    }

    let changed = false

    const insertAutoItem = (key, name, buildCard) => {
      if (existingAuto.has(key) || removed.has(key)) return
      const { tabNs, secNs } = splitNamespace(name)
      const tab = _ensureAutoTab(st, tabNs)
      const sec = _ensureAutoSection(tab, secNs)
      sec.cards.push(buildCard(tabNs))
      existingAuto.add(key)
      changed = true
    }

    for (const m of catalog.scalars || []) {
      insertAutoItem(`scalar:${m}`, m, () => ({
        id: `auto:scalar:${m}`,
        kind: "scalar",
        title: m,
        metrics: [m],
        widthPercent: st.defaultWidth,
        height: st.defaultHeight,
        custom: false,
      }))
    }

    for (const m of catalog.media || []) {
      insertAutoItem(`media:${m}`, m, () => ({
        id: `auto:media:${m}`,
        kind: "media",
        title: m,
        mediaName: m,
        currentStep: null,
        autoAdvance: true,
        widthPercent: st.defaultWidth,
        height: st.defaultHeight,
        custom: false,
      }))
    }

    for (const h of catalog.histograms || []) {
      insertAutoItem(`histogram:${h}`, h, (tabNs) => ({
        id: `auto:histogram:${h}`,
        kind: "distribution",
        distKind: "histogram",
        distName: h,
        title: h,
        histogramMode: tabNs === "gradients" || tabNs === "params" ? "flow" : "single",
        normalize: "per-step",
        currentStep: null,
        autoAdvance: true,
        widthPercent: st.defaultWidth,
        height: st.defaultHeight,
        custom: false,
      }))
    }

    for (const k of catalog.kdes || []) {
      insertAutoItem(`kde:${k}`, k, (tabNs) => ({
        id: `auto:kde:${k}`,
        kind: "distribution",
        distKind: "kde",
        distName: k,
        title: k,
        histogramMode: tabNs === "gradients" || tabNs === "params" ? "flow" : "single",
        normalize: "per-step",
        currentStep: null,
        autoAdvance: true,
        widthPercent: st.defaultWidth,
        height: st.defaultHeight,
        custom: false,
      }))
    }

    for (const tbl of catalog.tables || []) {
      insertAutoItem(`table:${tbl}`, tbl, () => ({
        id: `auto:table:${tbl}`,
        kind: "table",
        tableName: tbl,
        title: tbl,
        currentStep: null,
        autoAdvance: true,
        widthPercent: 50.0,
        height: st.defaultHeight,
        custom: false,
      }))
    }

    // Prune empty auto-placeholder root tab ("Metrics") once real namespace tabs exist
    if (st.tabs.length > 1) {
      const rootIdx = st.tabs.findIndex(
        (t) => t.id === "ns:_root" && t.autoNamespace === "" && countCardsInTab(t) === 0,
      )
      if (rootIdx !== -1) {
        st.tabs.splice(rootIdx, 1)
        changed = true
      }
    }

    // Guarantee at least one tab exists if catalog is completely empty
    if (st.tabs.length === 0) {
      const tab = createTab("Metrics", "")
      tab.sections.push(createSection("General", ""))
      st.tabs.push(tab)
      changed = true
    }

    if (!st.activeTabId || (!st.activeTabId.startsWith("builtin:") && !_findTab(st, st.activeTabId))) {
      st.activeTabId = st.tabs[0].id
      changed = true
    }

    if (changed) _save(scopeKey)
  }

  // ---- Tab CRUD ----

  function setActiveTab(scopeKey, tabId) {
    const st = ensure(scopeKey)
    st.activeTabId = tabId
    _save(scopeKey)
  }

  function addTab(scopeKey, name) {
    const st = ensure(scopeKey)
    const clean = (name || "").trim() || `Tab ${st.tabs.length + 1}`
    const tab = createTab(clean, null)
    tab.sections.push(createSection("General", null))
    st.tabs.push(tab)
    st.activeTabId = tab.id
    _save(scopeKey)
    return tab
  }

  function renameTab(scopeKey, tabId, newName) {
    const st = ensure(scopeKey)
    const tab = _findTab(st, tabId)
    const clean = (newName || "").trim()
    if (!tab || !clean) return
    tab.name = clean
    _save(scopeKey)
  }

  function removeTab(scopeKey, tabId) {
    const st = ensure(scopeKey)
    if (st.tabs.length <= 1) return
    const idx = st.tabs.findIndex((t) => t.id === tabId)
    if (idx === -1) return
    const [removed] = st.tabs.splice(idx, 1)
    for (const s of removed.sections) {
      for (const c of s.cards) {
        const k = autoKeyForCard(c)
        if (k && !st.removedKeys.includes(k)) st.removedKeys.push(k)
      }
    }
    if (st.activeTabId === tabId) {
      st.activeTabId = st.tabs[Math.min(idx, st.tabs.length - 1)]?.id || null
    }
    _save(scopeKey)
  }

  function reorderTab(scopeKey, draggedTabId, targetTabId) {
    const st = ensure(scopeKey)
    if (draggedTabId === targetTabId) return
    const from = st.tabs.findIndex((t) => t.id === draggedTabId)
    const to = st.tabs.findIndex((t) => t.id === targetTabId)
    if (from === -1 || to === -1) return
    const [moved] = st.tabs.splice(from, 1)
    st.tabs.splice(to, 0, moved)
    _save(scopeKey)
  }

  function updateTabSettings(scopeKey, tabId, patch) {
    const st = ensure(scopeKey)
    const tab = _findTab(st, tabId)
    if (!tab) return
    tab.settings = { ...tab.settings, ...patch }
    _save(scopeKey)
  }

  // ---- Section CRUD ----

  function addSection(scopeKey, tabId, title) {
    const st = ensure(scopeKey)
    const tab = _findTab(st, tabId)
    if (!tab) return null
    const clean = (title || "").trim() || `Section ${tab.sections.length + 1}`
    const sec = createSection(clean, null)
    tab.sections.push(sec)
    _save(scopeKey)
    return sec
  }

  function renameSection(scopeKey, tabId, sectionId, title) {
    const st = ensure(scopeKey)
    const sec = _findSection(_findTab(st, tabId), sectionId)
    const clean = (title || "").trim()
    if (!sec || !clean) return
    sec.title = clean
    _save(scopeKey)
  }

  function toggleSection(scopeKey, tabId, sectionId) {
    const st = ensure(scopeKey)
    const sec = _findSection(_findTab(st, tabId), sectionId)
    if (!sec) return
    sec.collapsed = !sec.collapsed
    _save(scopeKey)
  }

  function removeSection(scopeKey, tabId, sectionId) {
    const st = ensure(scopeKey)
    const tab = _findTab(st, tabId)
    if (!tab) return
    const idx = tab.sections.findIndex((s) => s.id === sectionId)
    if (idx === -1) return
    const [removed] = tab.sections.splice(idx, 1)
    // If other sections exist, preserve cards by moving them into the first remaining section
    if (removed.cards.length > 0 && tab.sections.length > 0) {
      tab.sections[0].cards.push(...removed.cards)
    } else {
      for (const c of removed.cards) {
        const k = autoKeyForCard(c)
        if (k && !st.removedKeys.includes(k)) st.removedKeys.push(k)
      }
    }
    if (tab.sections.length === 0) {
      tab.sections.push(createSection("General", null))
    }
    _save(scopeKey)
  }

  // ---- Card CRUD & Cross-Section / Cross-Tab Drag ----

  function addCard(scopeKey, tabId, sectionId, spec) {
    const st = ensure(scopeKey)
    const tab = _findTab(st, tabId)
    if (!tab) return null
    let sec = _findSection(tab, sectionId)
    if (!sec) {
      if (tab.sections.length === 0) tab.sections.push(createSection("General", null))
      sec = tab.sections[0]
    }
    const card = {
      id: uid("card"),
      widthPercent: spec.widthPercent ?? st.defaultWidth,
      height: spec.height ?? st.defaultHeight,
      custom: true,
      ...spec,
    }
    sec.cards.unshift(card)
    sec.collapsed = false
    _save(scopeKey)
    return card
  }

  function updateCard(scopeKey, tabId, sectionId, cardId, patch, { syncAll = false } = {}) {
    const st = ensure(scopeKey)
    const tab = _findTab(st, tabId)
    if (!tab) return
    if (syncAll) {
      for (const s of tab.sections) {
        for (const c of s.cards) {
          if (patch.widthPercent !== undefined) c.widthPercent = patch.widthPercent
          if (patch.height !== undefined) c.height = patch.height
        }
      }
      if (patch.widthPercent !== undefined) st.defaultWidth = patch.widthPercent
      if (patch.height !== undefined) st.defaultHeight = patch.height
      _save(scopeKey)
      return
    }
    const sec = _findSection(tab, sectionId)
    if (!sec) return
    const idx = sec.cards.findIndex((c) => c.id === cardId)
    if (idx === -1) return
    const oldHeight = sec.cards[idx].height
    const oldWidth = sec.cards[idx].widthPercent
    Object.assign(sec.cards[idx], patch)

    // OG v1 row-height alignment when resizing height (without width change)
    if (patch.height !== undefined && patch.height !== oldHeight && patch.widthPercent === undefined) {
      const rows = calculateRows(sec.cards)
      const row = rows.find((r) => r.includes(idx))
      if (row && row.length > 1) {
        for (const rIdx of row) sec.cards[rIdx].height = patch.height
      }
    } else if (patch.widthPercent !== undefined && patch.widthPercent !== oldWidth) {
      const rows = calculateRows(sec.cards)
      for (const row of rows) {
        if (row.length > 1) {
          const other = row.find((rIdx) => rIdx !== idx)
          if (other !== undefined) {
            const targetH = sec.cards[other].height
            for (const rIdx of row) sec.cards[rIdx].height = targetH
          }
        }
      }
    }
    _save(scopeKey)
  }

  function setTabCardsWidth(scopeKey, tabId, widthPercent) {
    const st = ensure(scopeKey)
    const tab = _findTab(st, tabId)
    if (!tab) return
    st.defaultWidth = widthPercent
    for (const s of tab.sections) {
      for (const c of s.cards) c.widthPercent = widthPercent
    }
    _save(scopeKey)
  }

  function removeCard(scopeKey, tabId, sectionId, cardId) {
    const st = ensure(scopeKey)
    const sec = _findSection(_findTab(st, tabId), sectionId)
    if (!sec) return
    const idx = sec.cards.findIndex((c) => c.id === cardId)
    if (idx === -1) return
    const [removed] = sec.cards.splice(idx, 1)
    const k = autoKeyForCard(removed)
    if (k && !st.removedKeys.includes(k)) st.removedKeys.push(k)
    _save(scopeKey)
  }

  /**
   * Move a card within a section, across sections, OR across Board Tabs (drag card -> drop on Tab pill!).
   */
  function moveCard(
    scopeKey,
    fromTabId,
    fromSectionId,
    cardId,
    toTabId,
    toSectionId = null,
    targetCardId = null,
  ) {
    const st = ensure(scopeKey)
    const srcTab = _findTab(st, fromTabId)
    const dstTab = _findTab(st, toTabId)
    if (!srcTab || !dstTab) return
    const srcSec = _findSection(srcTab, fromSectionId)
    if (!srcSec) return
    const fromIdx = srcSec.cards.findIndex((c) => c.id === cardId)
    if (fromIdx === -1) return

    let dstSec = toSectionId ? _findSection(dstTab, toSectionId) : dstTab.sections[0] || null
    if (!dstSec) {
      dstSec = createSection("General", null)
      dstTab.sections.push(dstSec)
    }

    if (srcSec === dstSec && targetCardId === cardId) return

    const [moved] = srcSec.cards.splice(fromIdx, 1)
    if (targetCardId) {
      const toIdx = dstSec.cards.findIndex((c) => c.id === targetCardId)
      if (toIdx !== -1) dstSec.cards.splice(toIdx, 0, moved)
      else dstSec.cards.push(moved)
    } else {
      dstSec.cards.push(moved)
    }
    dstSec.collapsed = false
    _save(scopeKey)
  }

  function saveAsTemplate(templateKey, scopeKey) {
    const st = ensure(scopeKey)
    if (typeof localStorage === "undefined") return
    try {
      localStorage.setItem(TEMPLATE_PREFIX + templateKey, JSON.stringify(st))
    } catch {
      // ignore storage quota error
    }
  }

  function resetBoard(scopeKey, catalog) {
    if (typeof localStorage !== "undefined") {
      localStorage.removeItem(STORAGE_PREFIX + scopeKey)
    }
    byScope.value[scopeKey] = defaultBoardState()
    syncCatalog(scopeKey, catalog)
  }

  function countCardsInTab(tab) {
    if (!tab?.sections) return 0
    let n = 0
    for (const s of tab.sections) n += s.cards.length
    return n
  }

  return {
    byScope,
    draggingCard,
    ensure,
    syncCatalog,
    setActiveTab,
    addTab,
    renameTab,
    removeTab,
    reorderTab,
    updateTabSettings,
    addSection,
    renameSection,
    toggleSection,
    removeSection,
    addCard,
    updateCard,
    setTabCardsWidth,
    removeCard,
    moveCard,
    saveAsTemplate,
    resetBoard,
    countCardsInTab,
  }
})
