export function formatNumber(val, digits = 4) {
  if (val === null || val === undefined || Number.isNaN(val)) return "NaN"
  if (!Number.isFinite(val)) return val > 0 ? "+Inf" : "-Inf"
  const abs = Math.abs(val)
  if (abs === 0) return "0"
  if (abs >= 1e6 || (abs > 0 && abs < 1e-3)) {
    return val.toExponential(Math.max(1, digits - 1))
  }
  return Number(val.toPrecision(digits)).toString()
}

export function formatCount(n) {
  if (n === null || n === undefined) return "-"
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}

export function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return "0 B"
  const units = ["B", "KiB", "MiB", "GiB"]
  let v = bytes
  let idx = 0
  while (v >= 1024 && idx < units.length - 1) {
    v /= 1024
    idx++
  }
  return idx === 0 ? `${Math.round(v)} B` : `${v.toFixed(1)} ${units[idx]}`
}

export function formatRelativeTime(ms) {
  if (!ms) return "-"
  const diff = Math.max(0, Date.now() - ms)
  const sec = Math.floor(diff / 1000)
  if (sec < 5) return "just now"
  if (sec < 60) return `${sec}s ago`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h ago`
  const days = Math.floor(hr / 24)
  return `${days}d ago`
}

export function formatDuration(startMs, endMs) {
  if (!startMs) return "-"
  const end = endMs || Date.now()
  const sec = Math.max(0, Math.floor((end - startMs) / 1000))
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = sec % 60
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}
