/**
 * Headless sharpness runner for the Python↔JS agreement test.
 * Usage: node --experimental-strip-types src/lib/sharpness_cli.ts case.json
 */
import { readFileSync } from 'node:fs'
import { fingertipSharpnessFromRgb } from './sharpness.ts'

const path = process.argv[2]
if (!path) {
  console.error('usage: sharpness_cli.ts <case.json>')
  process.exit(1)
}

type Case = {
  width: number
  height: number
  rgb: number[]
  tip: { x: number; y: number }
  dip: { x: number; y: number }
}

const data = JSON.parse(readFileSync(path, 'utf8')) as Case
const score = fingertipSharpnessFromRgb(
  data.rgb,
  data.width,
  data.height,
  data.tip,
  data.dip,
)
process.stdout.write(JSON.stringify({ sharpness: score }))
