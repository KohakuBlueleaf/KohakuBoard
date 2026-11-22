/**
 * Streaming binary frame decoder for `application/vnd.kobo.chunks`.
 *
 * Wire layout per frame:
 *   [header_len: u32le][header_json (UTF-8, padded so (4 + header_len) % 8 == 0)]
 *   [step: Float64Array(n)][global_step: Float64Array(n)][ts_ms: Float64Array(n)][value: Float64Array(n)]
 */

const textDecoder = new TextDecoder("utf-8")

function parseOneFrame(bytes, offset) {
  if (offset + 4 > bytes.byteLength) return null
  const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, 4)
  const headerLen = dv.getUint32(0, true)
  const payloadStart = offset + 4 + headerLen
  if (payloadStart > bytes.byteLength) return null

  const headerText = textDecoder.decode(bytes.subarray(offset + 4, payloadStart))
  const header = JSON.parse(headerText)
  const n = Math.max(0, header.end - header.start)
  const colBytes = n * 8
  const totalFrameBytes = 4 + headerLen + colBytes * 4
  if (offset + totalFrameBytes > bytes.byteLength) return null

  // Copy into a fresh aligned ArrayBuffer slice so Float64Array views never fault.
  const aligned = bytes.slice(payloadStart, payloadStart + colBytes * 4).buffer
  const step = new Float64Array(aligned, 0, n)
  const globalStep = new Float64Array(aligned, colBytes, n)
  const tsMs = new Float64Array(aligned, colBytes * 2, n)
  const value = new Float64Array(aligned, colBytes * 3, n)

  return {
    frame: { header, step, globalStep, tsMs, value },
    nextOffset: offset + totalFrameBytes,
  }
}

export function decodeSingleFrame(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer)
  const parsed = parseOneFrame(bytes, 0)
  return parsed ? parsed.frame : null
}

/**
 * Stream frames from a Fetch Response body and invoke `onFrame(frame)` immediately as each arrives.
 */
export async function consumeChunkStream(response, onFrame) {
  if (!response.body) {
    const buf = new Uint8Array(await response.arrayBuffer())
    let off = 0
    while (off < buf.byteLength) {
      const parsed = parseOneFrame(buf, off)
      if (!parsed) break
      onFrame(parsed.frame)
      off = parsed.nextOffset
    }
    return
  }

  const reader = response.body.getReader()
  let pending = new Uint8Array(0)

  while (true) {
    const { done, value } = await reader.read()
    if (value && value.byteLength > 0) {
      if (pending.byteLength === 0) {
        pending = value
      } else {
        const merged = new Uint8Array(pending.byteLength + value.byteLength)
        merged.set(pending, 0)
        merged.set(value, pending.byteLength)
        pending = merged
      }

      let off = 0
      while (off < pending.byteLength) {
        const parsed = parseOneFrame(pending, off)
        if (!parsed) break
        onFrame(parsed.frame)
        off = parsed.nextOffset
      }
      if (off > 0) {
        pending = off === pending.byteLength ? new Uint8Array(0) : pending.slice(off)
      }
    }
    if (done) break
  }
}
