# TooClose feature math

Spec for `src/tooclose/features.py` and the later TypeScript port.
All 3D math uses ordinary scalars (add, multiply, sqrt). No linear-algebra libraries.

Coordinate system is MediaPipe Hands:

- `x` in `[0, 1]` of image width, increasing to the right of the **processed** frame
- `y` in `[0, 1]` of image height, increasing downward
- `z` is wrist-relative depth, **smaller = closer to the camera**, roughly in units of image width

Pixel coordinates:

```
X = x * frame_width
Y = y * frame_height
Z = z * frame_width
```

## Handedness (physical hand)

`handedness` is the user's **physical** hand (`Left` / `Right`).

Phase 1 already flips MediaPipe's raw label after a selfie mirror (`hand_tracking.flip_handedness`).
`features.py` must **not** flip it again. It only uses the label to correct the palm-normal sign
(left and right hands are chiral — see below).

## 1. Focal length (pinhole)

Horizontal field of view `hfov_deg` is the angle subtended by `frame_width`.

```
hfov_rad = hfov_deg * π / 180
f_px = frame_width / (2 * tan(hfov_rad / 2))
```

`f_px` is the pinhole focal length in pixels.

## 2. Palm width and distance

Palm width is the image-plane distance between index MCP (landmark 5) and pinky MCP (17):

```
palm_width_px = hypot(X_index_mcp - X_pinky_mcp, Y_index_mcp - Y_pinky_mcp)
```

Assume a real palm width `P_cm` (default **8.5 cm**, `config.yaml` → `anatomy.palm_width_cm`).

Pinhole similar triangles:

```
distance_cm = (P_cm * f_px) / palm_width_px
```

If `palm_width_px < 1`, distance is missing.

This is the distance of the **MCP plane**, not of each fingertip. Fingers reaching toward
the lens are slightly closer; `dip_tip_norm` captures that per finger.

## 3. Pixels-per-inch at the fingertip

Assume a fingertip pad is `W_mm` wide (default **15 mm**).

At distance `D_cm`, 1 cm occupies `f_px / D_cm` pixels, so:

```
ppi = 2.54 * f_px / D_cm
```

Equivalent form using the 15 mm pad:

```
fingertip_px = (W_mm / 10) * f_px / D_cm    # 15 mm = 1.5 cm
ppi          = fingertip_px * 25.4 / W_mm
```

These are algebraically the same. TooClose stores `ppi` from `2.54 * f_px / D_cm`.

**Camera-dependence:** the same `D_cm` yields very different `ppi` on a 720p webcam vs a
4K phone, because `f_px` scales with resolution (and FOV). That is why resolution and
`hfov_deg` are features.

## 4. DIP→TIP length

For each finger, distal joint → tip in pixels, then normalized by frame width:

```
dip_tip_px   = hypot(X_tip - X_dip, Y_tip - Y_dip)
dip_tip_norm = dip_tip_px / frame_width
```

Thumb uses IP (landmark 3) in place of DIP.

## 5. Palm normal and pads-facing

3D points use `(X, Y, Z)` as defined above.

```
v_index = index_mcp - wrist     # landmarks 5 − 0
v_pinky = pinky_mcp - wrist     # landmarks 17 − 0
n_raw   = cross(v_index, v_pinky)
```

Cross product (right-handed):

```
cross(a, b) = (a.y*b.z - a.z*b.y,
               a.z*b.x - a.x*b.z,
               a.x*b.y - a.y*b.x)
```

**Chirality / handedness sign**

MediaPipe's frame is right-handed (`+x` right, `+y` down, `+z` away from the camera).

- **Right** hand, pads facing the camera: `n_raw` points **dorsal** (`+z`, into the scene).
- **Left** hand, pads facing the camera: `n_raw` points **palmar** (`−z`, toward the camera).

So the palmar normal (out of the finger pads) is:

```
n_palm = n_raw           if handedness == Left
n_palm = -n_raw          if handedness == Right
```

Toward-camera vector in this coordinate system:

```
c = (0, 0, -1)
```

```
pad_facing_cos       = dot(normalize(n_palm), c)     # +1 = pads straight at the lens
pad_facing_angle_deg = acos(clamp(pad_facing_cos, -1, 1)) * 180 / π
pads_facing_camera   = pad_facing_cos > 0            # angle < 90°
```

Selfie mirror: landmarks are expressed in the processed (usually mirrored) frame,
and `handedness` is already the physical hand. Horizontal flip reverses triangle
winding, so `extract_features` **un-flips landmark x** (`x → 1 − x`) before this
cross product. DIP→TIP pixel lengths are computed on the processed frame (mirror
does not change distances). Test: a palm-facing right hand must still be
`pads_facing_camera` after an x-flip with `mirrored=True`.

## 6. Finger curl

For index / middle / ring / pinky, angle at PIP between the two bone vectors:

```
v1 = pip - mcp
v2 = dip - pip
curl_deg = acos(clamp(dot(normalize(v1), normalize(v2)), -1, 1)) * 180 / π
```

- Fully extended (bones collinear, same direction): `curl_deg ≈ 0`
- Bent ~right angle: `curl_deg ≈ 90`

Thumb uses `(MCP, IP, TIP)` in place of `(MCP, PIP, DIP)`.

## 7. Sharpness (shared with TypeScript)

See also `src/tooclose/sharpness.py` and `frontend/src/lib/sharpness.ts`.

Crop exists only in memory and is discarded.

1. Distal length `L = dip_tip_px` (image pixels, not normalized).
2. Axis-aligned square centered on the tip, side `s = 1.5 * L`.
   Integer box: `x0 = floor(tip_x - s/2)`, `y0 = floor(tip_y - s/2)`,
   `side = max(1, round(s))`. Clip to the image; do not pad.
3. Grayscale, BT.601: `Y = 0.299 R + 0.587 G + 0.114 B` (OpenCV BGR input is reordered first).
4. Bilinear resize to 64×64. Output pixel `(i, j)` samples input at
   `x = (i + 0.5) * src_w / 64 - 0.5`, `y = (j + 0.5) * src_h / 64 - 0.5`,
   with edge clamp (no wrap).
5. 3×3 Laplacian kernel on interior pixels `(1..62, 1..62)`:

   ```
   0  1  0
   1 -4  1
   0  1  0
   ```

   `L = src[y-1][x] + src[y+1][x] + src[y][x-1] + src[y][x+1] - 4 * src[y][x]`
6. Return **population** variance of those 62×62 values:
   `var = mean(L²) - mean(L)²`.

Uniform patches → ~0. Sharp texture / focus → large values.

The browser supplies per-fingertip sharpness over the WebSocket; the server does not
see pixels. Python sharpness is for collection, debug, and the agreement test.

## 8. Feature dictionary (flat)

| key | meaning |
| --- | --- |
| `valid` | palm width was measurable |
| `handedness_right` | 1 if physical Right, else 0 |
| `confidence` | MediaPipe score |
| `frame_width`, `frame_height` | pixels |
| `hfov_deg` | horizontal FOV |
| `focal_length_px` | `f_px` |
| `palm_width_px`, `palm_width_norm` | MCP–MCP |
| `distance_cm` | pinhole estimate |
| `ppi` | at the MCP plane |
| `palm_normal_x/y/z` | unit palmar normal |
| `pad_facing_cos`, `pad_facing_angle_deg`, `pads_facing_camera` | see §5 |
| `{finger}_dip_tip_px`, `{finger}_dip_tip_norm` | §4 |
| `{finger}_curl_deg` | §6 |
| `{finger}_sharpness` | client-supplied or 0 |
| `mean_dip_tip_norm`, `max_dip_tip_norm`, `mean_sharpness` | summaries |
| `mirrored` | 1 if landmarks came from a selfie-mirrored frame |

Fingers: `thumb`, `index`, `middle`, `ring`, `pinky`.
