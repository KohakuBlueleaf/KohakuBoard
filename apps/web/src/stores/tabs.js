import { defineStore } from "pinia"
import { computed, ref } from "vue"
import { removeLeaf, splitLeaf, uid, updateRatio } from "@/utils/splitTree"

function groupLeaf(groupId) {
  return { id: groupId, kind: "leaf", groupId }
}

/**
 * KohakuTerrarium v2 TabGroup binary split tree store.
 *
 * Tabs live inside TabGroups. Dragging a tab to any edge of a group splits
 * the pane (VS Code style); closing the last tab in a split collapses it.
 */
export const useTabsStore = defineStore("tabs", () => {
  const initialGroupId = "grp_main"
  const groups = ref({
    [initialGroupId]: { id: initialGroupId, tabs: [], activeId: null },
  })
  const tree = ref(groupLeaf(initialGroupId))
  const focusedGroupId = ref(initialGroupId)
  const draggingTab = ref(null) // { tabId, fromGroupId }

  const groupCount = computed(() => Object.keys(groups.value).length)

  const allTabs = computed(() => {
    const out = []
    for (const g of Object.values(groups.value)) {
      for (const t of g.tabs) out.push(t)
    }
    return out
  })

  const focusedGroup = computed(
    () => groups.value[focusedGroupId.value] || Object.values(groups.value)[0] || null,
  )

  const activeTab = computed(() => {
    const g = focusedGroup.value
    if (!g) return null
    return g.tabs.find((t) => t.id === g.activeId) || g.tabs[0] || null
  })

  function findTabLocation(tabId) {
    for (const g of Object.values(groups.value)) {
      const idx = g.tabs.findIndex((t) => t.id === tabId)
      if (idx !== -1) return { groupId: g.id, index: idx, tab: g.tabs[idx] }
    }
    return null
  }

  function setFocusedGroup(groupId) {
    if (groups.value[groupId]) focusedGroupId.value = groupId
  }

  function _openOrFocus(tabSpec, targetGroupId = null) {
    const existing = findTabLocation(tabSpec.id)
    if (existing) {
      if (tabSpec.title) existing.tab.title = tabSpec.title
      groups.value[existing.groupId].activeId = existing.tab.id
      focusedGroupId.value = existing.groupId
      return existing.tab
    }
    const gid =
      (targetGroupId && groups.value[targetGroupId] ? targetGroupId : focusedGroupId.value) ||
      Object.keys(groups.value)[0]
    const group = groups.value[gid]
    group.tabs.push(tabSpec)
    group.activeId = tabSpec.id
    focusedGroupId.value = gid
    return tabSpec
  }

  function openProjectTab(project, targetGroupId = null) {
    return _openOrFocus(
      {
        id: `project:${project}`,
        kind: "project",
        project,
        runId: null,
        title: project,
        activeView: "charts",
      },
      targetGroupId,
    )
  }

  function openRunTab(project, runId, runName = null, targetGroupId = null) {
    return _openOrFocus(
      {
        id: `run:${project}/${runId}`,
        kind: "run",
        project,
        runId,
        title: runName || runId,
        activeView: "charts",
      },
      targetGroupId,
    )
  }

  function setTabView(tabId, view) {
    const loc = findTabLocation(tabId)
    if (loc) loc.tab.activeView = view
  }

  function activateTab(tabId, groupId = null) {
    const loc = groupId ? { groupId } : findTabLocation(tabId)
    if (!loc || !groups.value[loc.groupId]) return
    groups.value[loc.groupId].activeId = tabId
    focusedGroupId.value = loc.groupId
  }

  function _collapseEmptyGroup(groupId) {
    const g = groups.value[groupId]
    if (!g || g.tabs.length > 0 || Object.keys(groups.value).length <= 1) return
    delete groups.value[groupId]
    tree.value = removeLeaf(tree.value, groupId)
    if (focusedGroupId.value === groupId) {
      focusedGroupId.value = Object.keys(groups.value)[0]
    }
  }

  function closeTab(tabId) {
    const loc = findTabLocation(tabId)
    if (!loc) return
    const group = groups.value[loc.groupId]
    group.tabs.splice(loc.index, 1)
    if (group.activeId === tabId) {
      const next = group.tabs[Math.min(loc.index, group.tabs.length - 1)] || null
      group.activeId = next ? next.id : null
    }
    _collapseEmptyGroup(loc.groupId)
  }

  function reorderTab(groupId, draggedTabId, targetTabId) {
    const group = groups.value[groupId]
    if (!group || draggedTabId === targetTabId) return
    const fromIdx = group.tabs.findIndex((t) => t.id === draggedTabId)
    const toIdx = group.tabs.findIndex((t) => t.id === targetTabId)
    if (fromIdx === -1 || toIdx === -1) return
    const [moved] = group.tabs.splice(fromIdx, 1)
    group.tabs.splice(toIdx, 0, moved)
  }

  function moveTabToGroup(tabId, targetGroupId, insertIndex = -1) {
    const loc = findTabLocation(tabId)
    const dst = groups.value[targetGroupId]
    if (!loc || !dst) return
    if (loc.groupId === targetGroupId) {
      if (insertIndex >= 0 && insertIndex !== loc.index) {
        const [moved] = dst.tabs.splice(loc.index, 1)
        dst.tabs.splice(insertIndex, 0, moved)
      }
      dst.activeId = tabId
      focusedGroupId.value = targetGroupId
      return
    }
    const src = groups.value[loc.groupId]
    const [moved] = src.tabs.splice(loc.index, 1)
    if (src.activeId === tabId) {
      const next = src.tabs[Math.min(loc.index, src.tabs.length - 1)] || null
      src.activeId = next ? next.id : null
    }
    if (insertIndex >= 0 && insertIndex <= dst.tabs.length) {
      dst.tabs.splice(insertIndex, 0, moved)
    } else {
      dst.tabs.push(moved)
    }
    dst.activeId = moved.id
    focusedGroupId.value = targetGroupId
    _collapseEmptyGroup(loc.groupId)
  }

  function splitTabToEdge(tabId, targetGroupId, edge) {
    if (edge === "center") {
      moveTabToGroup(tabId, targetGroupId)
      return
    }
    const loc = findTabLocation(tabId)
    if (!loc || !groups.value[targetGroupId]) return
    // Don't split a solo tab against its own group.
    if (loc.groupId === targetGroupId && groups.value[targetGroupId].tabs.length <= 1) return

    const src = groups.value[loc.groupId]
    const [moved] = src.tabs.splice(loc.index, 1)
    if (src.activeId === tabId) {
      const next = src.tabs[Math.min(loc.index, src.tabs.length - 1)] || null
      src.activeId = next ? next.id : null
    }

    const newGroupId = uid("grp")
    groups.value[newGroupId] = { id: newGroupId, tabs: [moved], activeId: moved.id }

    const dir = edge === "left" || edge === "right" ? "h" : "v"
    const before = edge === "left" || edge === "top"
    tree.value = insertGroupSplit(tree.value, targetGroupId, newGroupId, dir, before)
    focusedGroupId.value = newGroupId
    _collapseEmptyGroup(loc.groupId)
  }

  function resizeSplit(splitId, ratio) {
    tree.value = updateRatio(tree.value, splitId, ratio)
  }

  return {
    groups,
    tree,
    focusedGroupId,
    draggingTab,
    groupCount,
    allTabs,
    focusedGroup,
    activeTab,
    setFocusedGroup,
    openProjectTab,
    openRunTab,
    setTabView,
    activateTab,
    closeTab,
    reorderTab,
    moveTabToGroup,
    splitTabToEdge,
    resizeSplit,
  }
})

function insertGroupSplit(root, targetGroupId, newGroupId, dir, before) {
  if (!root) return root
  if (root.kind === "leaf" && root.id === targetGroupId) {
    const a = before ? groupLeaf(newGroupId) : { ...root }
    const b = before ? { ...root } : groupLeaf(newGroupId)
    return { id: uid("ts"), kind: "split", dir, ratio: 50, a, b }
  }
  if (root.kind === "leaf") return root
  return {
    ...root,
    a: insertGroupSplit(root.a, targetGroupId, newGroupId, dir, before),
    b: insertGroupSplit(root.b, targetGroupId, newGroupId, dir, before),
  }
}
