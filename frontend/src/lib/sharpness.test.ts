import { describe, expect, it } from 'vitest'
import {
  fingertipSharpnessFromRgb,
  laplacianVariance,
  rgbToGrayBt601,
} from './sharpness'

function checkerRgb(width = 160, height = 120): number[] {
  const rgb: number[] = []
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const b = (x * 13 + y * 7) % 256
      const g = (x * 3 + y * 17) % 256
      const r = (x * y) % 256
      rgb.push(r, g, b)
    }
  }
  return rgb
}

describe('sharpness.ts', () => {
  it('uses BT.601 luma', () => {
    expect(rgbToGrayBt601(30, 20, 10)).toBeCloseTo(0.299 * 30 + 0.587 * 20 + 0.114 * 10)
  })

  it('returns 0 on a uniform crop', () => {
    const rgb = new Array(80 * 80 * 3).fill(128)
    const score = fingertipSharpnessFromRgb(rgb, 80, 80, { x: 40, y: 40 }, { x: 40, y: 20 })
    expect(score).toBeCloseTo(0, 8)
  })

  it('is large on the shared checker pattern', () => {
    const rgb = checkerRgb()
    const score = fingertipSharpnessFromRgb(rgb, 160, 120, { x: 80, y: 60 }, { x: 80, y: 20 })
    expect(score).toBeGreaterThan(100)
  })

  it('laplacian variance is 0 on a constant 64x64', () => {
    const gray = Array.from({ length: 64 }, () => Array(64).fill(10))
    expect(laplacianVariance(gray)).toBeCloseTo(0, 8)
  })
})
