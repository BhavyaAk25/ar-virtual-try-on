# 3D AR Jacket Try-On System

A professional-grade augmented reality application that overlays a realistic 3D jacket model on your body in real-time using computer vision, pose detection, and 3D rendering.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.8.1-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.8-orange.svg)
![PyRender](https://img.shields.io/badge/pyrender-latest-red.svg)

## ✨ Features

- **Real-time 3D Rendering**: Uses PyRender for professional Snapchat-filter-quality 3D jacket overlay
- **Adaptive Sizing**: Jacket automatically scales based on your body size and distance from camera
- **Face-Aware Positioning**: Uses facial landmark detection to ensure jacket never covers your face
- **Dynamic Movement Tracking**: Jacket follows your body movements in real-time with 30+ FPS
- **Perspective-Correct Rendering**: Proper 3D camera projection with lighting and depth
- **Universal Compatibility**: Works with any body size - adapts automatically
- **Scene-Aware Lighting**: Virtual lights match your room's brightness and color temperature so the 3D fabric stays readable
- **Live AR Control Center**: Hotkeys for scaling, vertical fit, style themes, and instant screenshots without touching the code
- **One-Tap Media Capture**: Letter-only shortcuts handle photos and full video recordings saved to the `captures/` folder

## 🎯 What Makes This Special

Unlike simple 2D image overlays, this system uses **true 3D rendering** with proper perspective projection, making it comparable to professional AR filters on Snapchat or Instagram.

## 📋 Requirements

- Python 3.7 or higher
- Webcam
- macOS, Windows, or Linux
- GPU recommended (but not required)

## 🚀 Installation

### Quick Start (macOS/Linux)

```bash
git clone https://github.com/BhavyaAk25/ar-virtual-try-on.git
cd ar-virtual-try-on
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python3 jacket_ar.py
```

### Manual Installation

1. **Clone and navigate to project directory:**
   ```bash
   git clone https://github.com/BhavyaAk25/ar-virtual-try-on.git
   cd ar-virtual-try-on
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install opencv-python mediapipe numpy trimesh pyrender
   ```

4. **Run the application:**
   ```bash
   python3 jacket_ar.py
   ```

## 🎮 Usage

### Controls

- `q`: Quit the application
- `a / d`: Shrink or grow the jacket scale in real time
- `w / s`: Move the jacket up or down to fine-tune vertical fit
- `c`: Cycle through high-contrast jacket materials
- `m`: Swap between GLB jacket models if multiple assets are present
- `r`: Reset tuning to the calibrated defaults
- `p`: Capture a photo (saved to `captures/`)
- `v`: Toggle MP4 video recording (saved next to the photos)
- Open palm (all 5 fingers visible): Trigger a hands-free color swap via MediaPipe Hands detection

### What to Expect

1. Application opens your webcam
2. Green skeleton overlay shows body tracking
3. Blue 3D jacket appears on your torso
4. Jacket automatically:
   - Scales to fit your body
   - Adjusts when you move closer/farther
   - Follows your movements
   - Stays below your face (doesn't cover eyes/mouth)
5. HUD text displays selected style, fit adjustments, and available hotkeys
6. Tap `p` or `v` to grab photos or MP4 recordings stored in `captures/`

### Best Results

- Stand 2-3 feet from camera
- Ensure good lighting
- Keep shoulders and hips visible in frame
- Face the camera directly

## 📁 Project Structure

```
ar-virtual-try-on/
├── jacket_ar.py           # Main AR application (3D rendering)
├── jacket.glb             # 3D jacket model (15MB)
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
├── README.md             # This file
├── PROJECT_OVERVIEW.md   # Technical architecture
├── QUICKSTART.md         # Quick start guide
├── TODO.md               # Development tasks
├── assets/               # Additional 3D model assets
├── logs/                 # Application logs
└── captures/             # Auto-created for AR photos & MP4 recordings
```

## 🔬 How It Works

### Technical Architecture

The system uses a multi-stage pipeline:

#### 1. Body Tracking
- **MediaPipe Pose** detects 33 body landmarks including:
  - Shoulders (for width measurement)
  - Hips (for torso height)
  - Mouth/nose (for face position)
- Green skeleton overlay shows tracking in real-time

#### 2. Adaptive Sizing (The Key Innovation)

**Problem We Solved:**
Initial attempts used fixed multipliers or complex perspective projection formulas that didn't adapt properly. The jacket was either too small, too large, or positioned incorrectly.

**Solution - Pixel-Based Adaptive Scaling:**

```python
# Measure body in pixels (same as green box tracking)
shoulder_width_pixels = distance(left_shoulder, right_shoulder)
torso_height_pixels = distance(avg_shoulders, avg_hips)

# Calculate scale relative to frame size
pixel_scale = max(shoulder_width_pixels, torso_height_pixels) / frame_height

# Depth adjustment: closer = bigger, farther = smaller
depth_factor = 1.0 / (1.0 + avg_z * 3.0)

# Final adaptive scale
scale = pixel_scale * depth_factor * 6.5
```

**Why This Works:**
- Uses actual pixel measurements (same system that draws the green box)
- Automatically adapts when you move closer/farther
- Works for any body size
- No complex camera calibration needed

#### 3. Face-Aware Positioning

**Problem We Solved:**
Initial positioning used torso center, which placed the jacket over the face.

**Solution - Mouth Tracking:**

```python
# Track mouth position
mouth_avg_px = average(mouth_left, mouth_right)

# Position jacket BELOW mouth
jacket_top_offset = torso_height_pixels * 0.5
jacket_center_y = mouth_y + offset + (jacket_height / 2)
```

This ensures the jacket collar starts below your chin, never covering eyes/nose/lips.

#### 4. 3D Rendering
- **PyRender** handles professional 3D rendering
- Perspective camera with 60° field of view
- Directional lighting for realistic appearance
- Alpha blending for smooth overlay

### The Journey: What Didn't Work

#### ❌ Attempt 1: Fixed Scale Multiplier
```python
scale = shoulder_width_3d * 25.0  # Fixed number
```
**Problem:** Didn't adapt to distance or body size changes.

#### ❌ Attempt 2: Complex Perspective Projection Math
```python
focal_length = h / (2 * tan(fov/2))
scale = (pixel_size * z_depth) / focal_length
```
**Problem:** Required accurate z_depth (which MediaPipe doesn't provide in absolute units). Math was theoretically correct but practically broken.

#### ✅ Attempt 3: Pixel-Based Adaptive System (Final Solution)
Uses relative measurements that automatically adapt. No camera calibration needed!

## 🤝 Collabratoin
-This project was done in collabration with Rudra Dasgupta (https://github.com/rudradasgupta7)

## 🎓 Key Learnings

### What Worked

1. **Pixel-based measurements** - More reliable than 3D coordinate math
2. **Relative depth factors** - MediaPipe's Z values work well for relative changes
3. **Facial landmark positioning** - Using mouth/nose to avoid face coverage
4. **Simple calibration constants** - 6.5x multiplier and 0.5 offset work universally

### What Didn't Work

1. **Fixed multipliers** - Don't adapt to movement
2. **Complex projection formulas** - Break without absolute depth measurements
3. **Torso-center positioning** - Covers the face
4. **Using only shoulder/hip landmarks** - Need facial landmarks for proper positioning

## 🛠️ Customization

### Adjust Jacket Size

Edit `jacket_ar.py` line 234:
```python
scale = pixel_scale * depth_factor * 6.5  # Change 6.5 to make bigger/smaller
```

- Increase (e.g., 7.5) = Bigger jacket
- Decrease (e.g., 5.5) = Smaller jacket
- Use the new `a / d` hotkeys to nudge this multiplier live without touching the file.

#### 3. Scene-Aware Lighting & Materials
- Each frame's torso ROI drives the virtual key light color + intensity so the jacket reflects the warmth/coolness of the real room.
- Ambient and rim lights are auto-balanced, which keeps the 3D folds visible even under dark or bright environments.
- Multiple MetallicRoughness material presets provide high-contrast looks without editing the GLB file.

#### 4. Live AR Control Center
- Fit multipliers (scale + torso offset) are exposed to hotkeys for instant tweaking while you're on camera.
- Style cycling swaps between cached PyRender meshes so color changes happen at 60 FPS.
- Screenshot hotkey writes PNGs to `captures/` so you can keep your favorite try-ons.

#### 5. Integrated Media Capture
- Letter-only shortcuts map to the most common actions: `p` for PNG photos, `v` to start/stop MP4 capture.
- Recordings use the live HUD output, so what you see in the preview is exactly what is saved.
- Everything lands in `captures/` for easy sharing.

### Adjust Jacket Position

Edit `jacket_ar.py` line 209:
```python
jacket_top_offset = torso_height_pixels * 0.5  # Change 0.5 to move up/down
```

- Increase (e.g., 0.6) = Lower position
- Decrease (e.g., 0.4) = Higher position
- In the live build you can tap `w` or `s` to raise/lower the torso offset instantly.

### Use Your Own 3D Model

Replace `jacket.glb` with your own GLB/GLTF file:
- Must be in GLB or GLTF format
- Model will be auto-normalized to unit size
- Works best with clothing/jacket models

## 🐛 Troubleshooting

### Jacket Not Visible
- Ensure shoulders and hips are visible in frame
- Check lighting conditions
- Make sure you're 2-3 feet from camera
- Verify `jacket.glb` exists in project directory

### Jacket Too Big/Small
- Adjust the `6.5` multiplier in line 234
- Move closer or farther from camera
- Check that green skeleton is tracking properly

### Jacket Covering Face
- Adjust the `0.5` offset in line 209 (increase to move jacket lower)
- Ensure facial landmarks are being detected (green dots on face)

### Low FPS / Laggy
- Close other applications
- Reduce camera resolution
- Disable skeleton drawing (comment out lines 165-169)
- Use lower MediaPipe model complexity

### Webcam Not Detected
- Close other apps using camera
- Check camera permissions
- Try different camera index (change `0` to `1` in line 97)

## 📊 Performance

- **FPS**: 30-60 FPS on modern computers
- **Latency**: 40-80ms total
- **Pose Detection**: 20-40ms
- **3D Rendering**: 10-30ms

## 🔧 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8.1.78 | Video capture and image processing |
| mediapipe | 0.10.8 | Real-time pose detection |
| numpy | 1.24.3 | Numerical operations |
| trimesh | latest | 3D model loading |
| pyrender | latest | Professional 3D rendering |

## 🚫 Important Notes

### Do NOT:
- Use PNG overlays for this project (we need 3D rendering)
- Rely on fixed scale multipliers
- Position based on torso center alone
- Ignore facial landmarks

### DO:
- Use the GLB 3D model
- Calculate scale from pixel measurements
- Position based on mouth landmarks
- Test with different body sizes and distances

## 🎯 Use Cases

- E-commerce virtual try-on
- Fashion retail AR experiences
- Social media filters
- Virtual fitting rooms
- Clothing design visualization


## 📝 Credits

Built with:
- **MediaPipe** (Google) - Body and facial landmark detection
- **PyRender** - Professional 3D rendering engine
- **Trimesh** - 3D model processing
- **OpenCV** - Computer vision library

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

