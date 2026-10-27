/**
 * Shared fingertip sharpness. Must match `src/tooclose/sharpness.py`
 * and `docs/feature_math.md` §7.
 *
 * 1. Crop a square around the tip, side = 1.5 × DIP→TIP (px).
 * 2. BT.601 grayscale: Y = 0.299 R + 0.587 G + 0.114 B.
 * 3. Bilinear resize to 64×64 (pixel-center mapping).
 * 4. 3×3 Laplacian [[0,1,0],[1,-4,1],[0,1,0]] on interior pixels.
 * 5. Population variance.
 *
 * Crops live only in memory.
 */

export const CROP_SCALE = 1.5
export const RESIZE = 64
export const BT601_R = 0.299
export const BT601_G = 0.587
export const BT601_B = 0.114

export type Point = { x: number; y: number }

function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value
}

export function rgbToGrayBt601(r: number, g: number, b: number): number {
  return BT601_R * r + BT601_G * g + BT601_B * b
}

export function sampleBilinear(gray: number[][], x: number, y: number): number {
  const height = gray.length
  const width = height ? gray[0].length : 0
  if (height === 0 || width === 0) return 0
  const xClamped = clamp(x, 0, width - 1)
  const yClamped = clamp(y, 0, height - 1)
  const x0 = Math.floor(xClamped)
  const y0 = Math.floor(yClamped)
  const x1 = Math.min(x0 + 1, width - 1)
  const y1 = Math.min(y0 + 1, height - 1)
  const tx = xClamped - x0
  const ty = yClamped - y0
  const top = gray[y0][x0] * (1 - tx) + gray[y0][x1] * tx
  const bot = gray[y1][x0] * (1 - tx) + gray[y1][x1] * tx
  return top * (1 - ty) + bot * ty
}

export function resizeGrayBilinear(
  gray: number[][],
  outW = RESIZE,
  outH = RESIZE,
): number[][] {
  const srcH = gray.length
  const srcW = srcH ? gray[0].length : 0
  if (srcH === 0 || srcW === 0) {
    return Array.from({ length: outH }, () => Array(outW).fill(0))
  }
  const out: number[][] = []
  for (let j = 0; j < outH; j++) {
    const y = ((j + 0.5) * srcH) / outH - 0.5
    const row: number[] = []
    for (let i = 0; i < outW; i++) {
      const x = ((i + 0.5) * srcW) / outW - 0.5
      row.push(sampleBilinear(gray, x, y))
    }
    out.push(row)
  }
  return out
}

export function laplacianVariance(gray64: number[][]): number {
  const height = gray64.length
  const width = height ? gray64[0].length : 0
  if (height < 3 || width < 3) return 0
  let total = 0
  let totalSq = 0
  let count = 0
  for (let y = 1; y < height - 1; y++) {
    const row = gray64[y]
    const rowUp = gray64[y - 1]
    const rowDn = gray64[y + 1]
    for (let x = 1; x < width - 1; x++) {
      const value = rowUp[x] + rowDn[x] + row[x - 1] + row[x + 1] - 4 * row[x]
      total += value
      totalSq += value * value
      count += 1
    }
  }
  if (count === 0) return 0
  const mean = total / count
  return totalSq / count - mean * mean
}

/** Packed RGB bytes (R,G,B repeating), row-major. Used by tests and the canvas path. */
export function cropSquareGrayRgb(
  rgb: Uint8Array | number[],
  width: number,
  height: number,
  cx: number,
  cy: number,
  side: number,
): number[][] {
  const sidePx = Math.max(1, Math.round(side))
  const x0 = Math.floor(cx - side / 2)
  const y0 = Math.floor(cy - side / 2)
  const x1 = x0 + sidePx
  const y1 = y0 + sidePx
  const x0c = Math.max(0, x0)
  const y0c = Math.max(0, y0)
  const x1c = Math.min(width, x1)
  const y1c = Math.min(height, y1)
  if (x1c <= x0c || y1c <= y0c) return []
  const patch: number[][] = []
  for (let y = y0c; y < y1c; y++) {
    const row: number[] = []
    for (let x = x0c; x < x1c; x++) {
      const i = (y * width + x) * 3
      row.push(rgbToGrayBt601(rgb[i], rgb[i + 1], rgb[i + 2]))
    }
    patch.push(row)
  }
  return patch
}

export function fingertipSharpnessFromRgb(
  rgb: Uint8Array | number[],
  width: number,
  height: number,
  tip: Point,
  dip: Point,
  cropScale = CROP_SCALE,
  resize = RESIZE,
): number {
  const dipTip = Math.hypot(tip.x - dip.x, tip.y - dip.y)
  const side = cropScale * dipTip
  if (side < 1) return 0
  const gray = cropSquareGrayRgb(rgb, width, height, tip.x, tip.y, side)
  if (gray.length === 0) return 0
  return laplacianVariance(resizeGrayBilinear(gray, resize, resize))
}

/**
 * Browser helper: read an already-drawn video/canvas as RGB and run the shared formula.
 * Offscreen canvas stays in memory and is discarded with the function scope.
 */
export function fingertipSharpness(
  source: CanvasImageSource & { width?: number; height?: number },
  tip: Point,
  dip: Point,
  frameWidth: number,
  frameHeight: number,
): number {
  const canvas = document.createElement('canvas')
  canvas.width = frameWidth
  canvas.height = frameHeight
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return 0
  ctx.drawImage(source, 0, 0, frameWidth, frameHeight)
  const image = ctx.getImageData(0, 0, frameWidth, frameHeight)
  const rgb = new Uint8Array(frameWidth * frameHeight * 3)
  for (let p = 0, q = 0; p < image.data.length; p += 4, q += 3) {
    rgb[q] = image.data[p]
    rgb[q + 1] = image.data[p + 1]
    rgb[q + 2] = image.data[p + 2]
  }
  return fingertipSharpnessFromRgb(rgb, frameWidth, frameHeight, tip, dip)
}
