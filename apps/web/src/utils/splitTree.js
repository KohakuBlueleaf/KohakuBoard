/**
 * Pure binary split-tree helpers for KohakuTerrarium-style WorkspaceShell layouts.
 */

let nextId = 1
export function uid(prefix = "n") {
  return `${prefix}_${nextId++}_${Math.random().toString(36).slice(2, 6)}`
}

export function leaf(panel) {
  return { id: uid("leaf"), kind: "leaf", panel }
}

export function hsplit(ratio, a, b) {
  return { id: uid("hs"), kind: "split", dir: "h", ratio, a, b }
}

export function vsplit(ratio, a, b) {
  return { id: uid("vs"), kind: "split", dir: "v", ratio, a, b }
}

export function cloneTree(node) {
  if (!node) return null
  if (node.kind === "leaf") return { ...node }
  return { ...node, a: cloneTree(node.a), b: cloneTree(node.b) }
}

export function replaceNode(root, targetId, replacement) {
  if (!root) return root
  if (root.id === targetId) return replacement
  if (root.kind === "leaf") return root
  return {
    ...root,
    a: replaceNode(root.a, targetId, replacement),
    b: replaceNode(root.b, targetId, replacement),
  }
}

export function splitLeaf(root, targetId, dir, newPanel) {
  if (!root) return root
  if (root.id === targetId && root.kind === "leaf") {
    const maker = dir === "h" ? hsplit : vsplit
    return maker(50, { ...root }, leaf(newPanel))
  }
  if (root.kind === "leaf") return root
  return {
    ...root,
    a: splitLeaf(root.a, targetId, dir, newPanel),
    b: splitLeaf(root.b, targetId, dir, newPanel),
  }
}

export function removeLeaf(root, targetId) {
  if (!root || root.kind === "leaf") return root
  if (root.a.id === targetId) return root.b
  if (root.b.id === targetId) return root.a
  return {
    ...root,
    a: removeLeaf(root.a, targetId),
    b: removeLeaf(root.b, targetId),
  }
}

export function updateRatio(root, splitId, ratio) {
  if (!root || root.kind === "leaf") return root
  const clamped = Math.max(15, Math.min(85, Math.round(ratio)))
  if (root.id === splitId) return { ...root, ratio: clamped }
  return {
    ...root,
    a: updateRatio(root.a, splitId, clamped),
    b: updateRatio(root.b, splitId, clamped),
  }
}

export function countLeaves(root) {
  if (!root) return 0
  if (root.kind === "leaf") return 1
  return countLeaves(root.a) + countLeaves(root.b)
}
