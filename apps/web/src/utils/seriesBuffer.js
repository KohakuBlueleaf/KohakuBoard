/**
 * Pre-allocated tail-first Float64Array buffer for newest-first progressive full-data loading.
 *
 * Eliminates O(N^2) array concatenation while older chunks stream backward from N -> 0.
 */

export class SeriesBuffer {
  constructor(totalRows) {
    this.total = Math.max(0, totalRows)
    this.capacity = Math.max(this.total, 1024)
    this.step = new Float64Array(this.capacity)
    this.globalStep = new Float64Array(this.capacity)
    this.tsMs = new Float64Array(this.capacity)
    this.value = new Float64Array(this.capacity)
    this.loadedStart = this.total
    this.loadedEnd = this.total
    this.hasNonFinite = false
    this.version = 0
  }

  _ensureCapacity(needed) {
    if (needed <= this.capacity) return
    let nextCap = this.capacity
    while (nextCap < needed) nextCap *= 2
    const grow = (old) => {
      const next = new Float64Array(nextCap)
      next.set(old)
      return next
    }
    this.step = grow(this.step)
    this.globalStep = grow(this.globalStep)
    this.tsMs = grow(this.tsMs)
    this.value = grow(this.value)
    this.capacity = nextCap
  }

  writeChunk(start, end, stepArr, gstepArr, tsArr, valArr, totalRows = null) {
    if (totalRows !== null && totalRows > this.total) {
      this._ensureCapacity(totalRows)
      this.total = totalRows
    }
    this._ensureCapacity(end)
    this.step.set(stepArr, start)
    this.globalStep.set(gstepArr, start)
    this.tsMs.set(tsArr, start)
    this.value.set(valArr, start)
    for (let i = 0; i < valArr.length; i++) {
      if (!Number.isFinite(valArr[i])) {
        this.hasNonFinite = true
        break
      }
    }
    if (this.loadedStart === this.loadedEnd) {
      this.loadedStart = start
      this.loadedEnd = end
    } else {
      if (start < this.loadedStart) this.loadedStart = start
      if (end > this.loadedEnd) this.loadedEnd = end
    }
    this.version++
  }

  get loadedCount() {
    return Math.max(0, this.loadedEnd - this.loadedStart)
  }

  get progress() {
    if (this.total === 0) return 1
    return Math.min(1, this.loadedCount / this.total)
  }

  get complete() {
    return this.loadedStart === 0
  }

  view(xAxis = "global_step") {
    const a = this.loadedStart
    const b = this.loadedEnd
    let xSrc = this.globalStep
    if (xAxis === "step") xSrc = this.step
    else if (xAxis === "relative_time") {
      const t = this.tsMs.subarray(a, b)
      const rel = new Float64Array(t.length)
      const origin = t.length > 0 ? t[0] : 0
      for (let i = 0; i < t.length; i++) {
        rel[i] = (t[i] - origin) / 1000.0
      }
      return {
        x: rel,
        y: this.value.subarray(a, b),
        loaded: b - a,
        total: this.total,
        progress: this.progress,
        complete: this.complete,
        hasNonFinite: this.hasNonFinite,
      }
    }
    return {
      x: xSrc.subarray(a, b),
      y: this.value.subarray(a, b),
      loaded: b - a,
      total: this.total,
      progress: this.progress,
      complete: this.complete,
      hasNonFinite: this.hasNonFinite,
    }
  }
}

/**
 * Pair an arbitrary scalar X-metric buffer against a Y-metric buffer by matching step.
 */
export function pairMetricAxes(xBuf, yBuf) {
  if (!xBuf || !yBuf || xBuf.loadedCount === 0 || yBuf.loadedCount === 0) {
    return { x: new Float64Array(0), y: new Float64Array(0) }
  }
  const xMap = new Map()
  for (let i = xBuf.loadedStart; i < xBuf.loadedEnd; i++) {
    const xv = xBuf.value[i]
    if (Number.isFinite(xv)) xMap.set(xBuf.step[i], xv)
  }
  const xs = []
  const ys = []
  for (let i = yBuf.loadedStart; i < yBuf.loadedEnd; i++) {
    const xv = xMap.get(yBuf.step[i])
    if (xv !== undefined) {
      xs.push(xv)
      ys.push(yBuf.value[i])
    }
  }
  return { x: Float64Array.from(xs), y: Float64Array.from(ys) }
}

/**
 * Bias-corrected EMA smoothing over a Float64Array (preserving NaN gaps).
 */
export function smoothEma(values, weight) {
  if (!weight || weight <= 0) return values
  const alpha = Math.max(0.001, Math.min(0.999, weight))
  const out = new Float64Array(values.length)
  let acc = 0
  let denom = 0
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) {
      out[i] = Number.NaN
      continue
    }
    acc = acc * alpha + v * (1 - alpha)
    denom = denom * alpha + (1 - alpha)
    out[i] = denom > 0 ? acc / denom : v
  }
  return out
}

/**
 * Centered Moving Average smoothing.
 */
export function smoothMovingAverage(values, windowSize) {
  const w = Math.max(1, Math.round(windowSize || 1))
  if (w <= 1) return values
  const half = Math.floor(w / 2)
  const n = values.length
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(values[i])) {
      out[i] = Number.NaN
      continue
    }
    const lo = Math.max(0, i - half)
    const hi = Math.min(n - 1, i + half)
    let sum = 0
    let cnt = 0
    for (let j = lo; j <= hi; j++) {
      const v = values[j]
      if (Number.isFinite(v)) {
        sum += v
        cnt++
      }
    }
    out[i] = cnt > 0 ? sum / cnt : Number.NaN
  }
  return out
}

/**
 * 1D Gaussian kernel smoothing (sigma = window / 3).
 */
export function smoothGaussian(values, sigma) {
  const s = Math.max(0.5, Number(sigma) || 1)
  const radius = Math.min(64, Math.max(1, Math.ceil(s * 3)))
  const kernel = new Float64Array(radius * 2 + 1)
  const inv2s2 = 1 / (2 * s * s)
  for (let k = -radius; k <= radius; k++) {
    kernel[k + radius] = Math.exp(-k * k * inv2s2)
  }
  const n = values.length
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(values[i])) {
      out[i] = Number.NaN
      continue
    }
    let sum = 0
    let wsum = 0
    const lo = Math.max(0, i - radius)
    const hi = Math.min(n - 1, i + radius)
    for (let j = lo; j <= hi; j++) {
      const v = values[j]
      if (Number.isFinite(v)) {
        const kw = kernel[j - i + radius]
        sum += v * kw
        wsum += kw
      }
    }
    out[i] = wsum > 0 ? sum / wsum : Number.NaN
  }
  return out
}

export function applySmoothing(values, mode = "none", value = 0) {
  if (!mode || mode === "none" || !value || value <= 0) return null
  if (mode === "ema") return smoothEma(values, value)
  if (mode === "ma") return smoothMovingAverage(values, value)
  if (mode === "gaussian") return smoothGaussian(values, value)
  return null
}

/**
 * Align multiple `(x, y)` Float64Array series onto a single sorted union X axis for uPlot.
 */
export function alignSeriesForUPlot(entries, opts = {}) {
  const mode = typeof opts === "number" ? (opts > 0 ? "ema" : "none") : opts.mode || "none"
  const value = typeof opts === "number" ? opts : Number(opts.value || 0)
  const showOriginal = typeof opts === "number" ? true : opts.showOriginal !== false
  const hasSmooth = mode !== "none" && value > 0

  if (entries.length === 0) return [[]]
  if (entries.length === 1 && !hasSmooth) {
    const { x, y } = entries[0]
    // Ensure monotonic X for uPlot
    let sorted = true
    for (let i = 1; i < x.length; i++) {
      if (x[i] < x[i - 1]) {
        sorted = false
        break
      }
    }
    if (sorted) return [x, sanitizeForUPlot(y)]
  }

  const xSet = new Set()
  for (const e of entries) {
    for (let i = 0; i < e.x.length; i++) {
      const xv = e.x[i]
      if (Number.isFinite(xv)) xSet.add(xv)
    }
  }
  const mergedX = Float64Array.from(xSet)
  mergedX.sort()

  const xIndex = new Map()
  for (let i = 0; i < mergedX.length; i++) {
    xIndex.set(mergedX[i], i)
  }

  const cols = [mergedX]
  for (const e of entries) {
    const rawCol = new Array(mergedX.length).fill(null)
    const smoothSource = hasSmooth ? applySmoothing(e.y, mode, value) : null
    const smoothCol = smoothSource ? new Array(mergedX.length).fill(null) : null

    for (let i = 0; i < e.x.length; i++) {
      const idx = xIndex.get(e.x[i])
      if (idx === undefined) continue
      const rv = e.y[i]
      rawCol[idx] = Number.isFinite(rv) ? rv : null
      if (smoothCol && smoothSource) {
        const sv = smoothSource[i]
        smoothCol[idx] = Number.isFinite(sv) ? sv : null
      }
    }
    if (smoothCol) {
      cols.push(smoothCol)
      if (showOriginal) cols.push(rawCol)
    } else {
      cols.push(rawCol)
    }
  }
  return cols
}

function sanitizeForUPlot(arr) {
  const out = new Array(arr.length)
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]
    out[i] = Number.isFinite(v) ? v : null
  }
  return out
}
