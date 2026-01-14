# Trail Project TODO

- [x] Implement scene-aware lighting so the 3D jacket matches the room's brightness and color temperature (keeps rendering readable in any lighting).
- [x] Build live AR controls: runtime scale/offset tuning, color themes, and a screenshot hotkey so users can experiment without editing code.
- [x] Add all-letter hotkeys plus MP4 recording toggle for easier media capture.

## Next Up (easiest ➝ hardest)

1. [x] Add a diagnostics overlay + structured logging so FPS, pose confidence, and render timings are visible live and persisted to disk for tuning.
2. [x] Build a style manager that can load multiple GLB jackets from a `assets/jackets/` folder, expose metadata (name, color preset), and let users cycle models live.
3. [x] Implement depth-aware occlusion using MediaPipe segmentation + depth heuristics so the jacket can pass behind arms/props instead of always overpainting.
