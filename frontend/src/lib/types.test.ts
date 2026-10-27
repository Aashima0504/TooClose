import { describe, expect, it } from 'vitest'
import type { ClientMessage } from './types'

describe('client message shape', () => {
  it('documents the privacy contract: no frame payload field', () => {
    const keys: (keyof ClientMessage)[] = [
      'landmarks',
      'handedness',
      'confidence',
      'frame',
      'camera',
      'sharpness',
    ]
    expect(keys).not.toContain('image')
    expect(keys).not.toContain('frameData')
  })
})
