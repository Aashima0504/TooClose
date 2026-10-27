export type Handedness = 'Left' | 'Right'
export type CameraType = 'webcam' | 'phone'
export type RiskState = 'safe' | 'warning' | 'danger'

export type Landmark = [number, number, number]

export interface FrameSize {
  width: number
  height: number
}

export interface CameraSettings {
  type: CameraType
  hfov_deg: number
}

export interface SharpnessValues {
  thumb: number
  index: number
  middle: number
  ring: number
  pinky: number
}

/** Client -> server. Landmarks only — never a frame. */
export interface ClientMessage {
  landmarks: Landmark[]
  handedness: Handedness
  confidence: number
  frame: FrameSize
  camera: CameraSettings
  sharpness: SharpnessValues
}

export interface ShapReason {
  feature: string
  label: string
  impact: number
}

export interface DebugInfo {
  distance_cm: number
  ppi: number
  sharpness: number
  pad_facing_angle: number
  latency_ms: number
}

/** Server -> client. */
export interface ServerMessage {
  risk: number
  state: RiskState
  reasons: ShapReason[]
  advice: string
  debug: DebugInfo
}
