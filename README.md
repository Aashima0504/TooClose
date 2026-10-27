# TooClose

Local, real-time warning when fingertips are close enough to a camera that
fingerprints could be captured.

**Status:** Phase 1 scaffold. Video frames stay in this process (debug view) or
in the browser (later phases). They are never written to disk and never sent to
a remote server.

Browsers only allow webcam access on **localhost** or **HTTPS**.

## Phase 1 — run and verify

Python **3.11** is recommended (MediaPipe still lags on 3.13).

```bash
make setup          # venv, pip, MediaPipe .task model, npm install
make test           # config + /health smoke tests
make debug          # OpenCV landmark viewer (q quit, m toggle mirror)
make dev            # FastAPI on 127.0.0.1:8000 + Vite on localhost:5173
```

- `GET http://127.0.0.1:8000/health` should return `{"status":"ok",...}`
- `GET http://127.0.0.1:8000/config` should return the public config subset
- Debug view should draw a 21-point skeleton when a hand is in frame
- No JPEG/PNG files should appear under `data/` or the repo root

Full architecture, threat model, results, and privacy design land in Phase 10.
