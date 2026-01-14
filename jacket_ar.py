"""
PROFESSIONAL 3D JACKET AR SYSTEM
Uses PyRender for proper 3D rendering like Snapchat filters
Expert-level computer vision implementation
"""

import csv
import json
import cv2
import mediapipe as mp
import numpy as np
import trimesh
import pyrender
import time
import os
from datetime import datetime

print("\n" + "="*60)
print("PROFESSIONAL 3D JACKET AR")
print("Expert CV System - Like Snapchat Filters")
print("="*60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JACKET_ASSET_DIR = os.path.join(BASE_DIR, "assets", "jackets")
DEFAULT_STYLE_COLOR = [0.20, 0.35, 0.95, 1.0]
BASE_SCALE_CALIBRATION = 6.5
BASE_HEIGHT_OFFSET = 0.5
os.makedirs(JACKET_ASSET_DIR, exist_ok=True)

COLOR_VARIANT_LIBRARY = [
    {"name": "Studio Navy", "color": [0.20, 0.35, 0.95, 1.0]},
    {"name": "Matte Black", "color": [0.08, 0.08, 0.08, 1.0]},
    {"name": "Neon Ember", "color": [0.90, 0.25, 0.20, 1.0]},
    {"name": "Arctic Cyan", "color": [0.25, 0.90, 0.95, 1.0]},
    {"name": "Sunset Copper", "color": [0.95, 0.55, 0.25, 1.0]}
]


def ensure_rgba(color_values):
    """Guarantee colors are RGBA lists."""
    if color_values is None:
        return DEFAULT_STYLE_COLOR[:]
    if isinstance(color_values, dict):
        extracted = color_values.get("color") or color_values.get("rgba") or color_values.get("value")
        if extracted is None:
            return DEFAULT_STYLE_COLOR[:]
        color_values = extracted
    rgba = list(color_values)
    if len(rgba) == 3:
        rgba.append(1.0)
    if len(rgba) != 4:
        return DEFAULT_STYLE_COLOR[:]
    return [float(max(0.0, min(1.0, c))) for c in rgba]


def build_color_variants():
    """Return the five calibrated color cycles used by hotkeys and gestures."""
    return [
        {"name": preset["name"], "color": ensure_rgba(preset["color"])}
        for preset in COLOR_VARIANT_LIBRARY
    ]


def load_mesh_from_path(path: str, apply_scene_graph: bool = False) -> trimesh.Trimesh:
    """Load, merge, and normalize a GLB/GLTF file into a unit-sized mesh."""
    scene_or_mesh = trimesh.load(path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = []
        if apply_scene_graph:
            for node_name in scene_or_mesh.graph.nodes_geometry:
                transform, geometry_name = scene_or_mesh.graph[node_name]
                geom = scene_or_mesh.geometry.get(geometry_name)
                if isinstance(geom, trimesh.Trimesh):
                    transformed = geom.copy()
                    transformed.apply_transform(transform)
                    meshes.append(transformed)
        else:
            for geom in scene_or_mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom.copy())
        if not meshes:
            raise ValueError(f"No mesh geometries found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    largest_axis = max(size)
    if largest_axis == 0:
        raise ValueError(f"Mesh {path} has zero size after normalization")
    mesh.vertices /= largest_axis
    return mesh


def load_style_assets():
    """Load all jacket styles from the manifest or fallback to the default asset."""
    manifest_path = os.path.join(JACKET_ASSET_DIR, "manifest.json")
    manifest_entries = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                data = json.load(manifest_file)
                if isinstance(data, list):
                    manifest_entries = data
                else:
                    print("   ⚠️ Manifest must be a list - ignoring.")
        except Exception as manifest_error:
            print(f"   ⚠️ Failed to read manifest.json ({manifest_error}). Using fallback.")

    if not manifest_entries:
        manifest_entries = [{
            "file": "jacket.glb",
            "name": "Studio Tailor (fallback)",
            "color": DEFAULT_STYLE_COLOR
        }]

    styles = []
    for entry in manifest_entries:
        asset_file = entry.get("file", "").strip()
        if not asset_file:
            continue
        asset_path = asset_file if os.path.isabs(asset_file) else os.path.join(JACKET_ASSET_DIR, asset_file)
        if not os.path.exists(asset_path):
            fallback_path = os.path.join(BASE_DIR, asset_file)
            if os.path.exists(fallback_path):
                asset_path = fallback_path
            else:
                print(f"   ⚠️ Missing asset {asset_file}, skipping entry.")
                continue
        apply_graph = bool(entry.get("metadata", {}).get("apply_scene_graph", False))
        try:
            mesh = load_mesh_from_path(asset_path, apply_scene_graph=apply_graph)
            styles.append({
                "name": entry.get("name") or os.path.splitext(os.path.basename(asset_path))[0].replace("_", " ").title(),
                "color": entry.get("color", DEFAULT_STYLE_COLOR),
                "file": asset_path,
                "metadata": entry.get("metadata", {}),
                "mesh": mesh,
                "color_variants": build_color_variants(),
                "current_color_idx": 0,
                "fit_defaults": {
                    "scale": float(entry.get("metadata", {}).get("fit", {}).get("scale", BASE_SCALE_CALIBRATION)),
                    "offset": float(entry.get("metadata", {}).get("fit", {}).get("offset", BASE_HEIGHT_OFFSET))
                }
            })
        except Exception as mesh_error:
            print(f"   ⚠️ Failed to load {asset_path}: {mesh_error}")

    return styles

# ============================================================================
# STEP 1: Load jacket assets
# ============================================================================
print("\n[1/6] Loading jacket assets...")
style_assets = load_style_assets()

if not style_assets:
    print("   ❌ No jacket styles could be loaded. Please add GLB files to assets/jackets.")
    exit(1)

for style in style_assets:
    verts = len(style["mesh"].vertices)
    faces = len(style["mesh"].faces)
    print(f"   ✅ {style['name']} ({os.path.basename(style['file'])}) - {verts} verts / {faces} faces")

mesh_cache = {}
current_style_idx = 0
current_style_meta = style_assets[current_style_idx]
current_color_variant_name = current_style_meta["color_variants"][current_style_meta["current_color_idx"]]["name"]
style_default_scale = current_style_meta["fit_defaults"]["scale"]
style_default_offset = current_style_meta["fit_defaults"]["offset"]


def build_jacket_mesh(style_idx: int) -> pyrender.Mesh:
    """Create (or fetch cached) PyRender mesh for a specific style asset."""
    style = style_assets[style_idx % len(style_assets)]
    variants = style["color_variants"]
    color_idx = style["current_color_idx"] % len(variants)
    color_variant = variants[color_idx]
    cache_key = f"{style['name']}::{color_variant['name']}"

    if cache_key in mesh_cache:
        return mesh_cache[cache_key]

    material_meta = style.get("metadata", {}).get("material", {})
    color = ensure_rgba(color_variant["color"])

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color,
        metallicFactor=material_meta.get("metallic", 0.35),
        roughnessFactor=material_meta.get("roughness", 0.5),
        emissiveFactor=material_meta.get("emissive", [c * 0.05 for c in color[:3]])
    )
    mesh = pyrender.Mesh.from_trimesh(style["mesh"], material=material, smooth=True)
    mesh_cache[cache_key] = mesh
    return mesh

# ============================================================================
# STEP 2: Initialize PyRender (Professional 3D renderer)
# ============================================================================
print("\n[2/6] Initializing PyRender 3D engine...")
try:
    # Create pyrender mesh from selected style
    pr_jacket = build_jacket_mesh(current_style_idx)

    # Create rendering scene
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[0, 0, 0, 0])

    # Add directional light (like sunlight)
    main_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.2)
    main_light_node = scene.add(main_light, pose=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ]))

    rim_light = pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=1.5)
    rim_light_node = scene.add(rim_light, pose=np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, -2],
        [0, 0, 0, 1]
    ]))

    # Create perspective camera (matches human vision)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    print(f"   ✅ PyRender initialized with lighting")
    print(f"   🎨 Active style: {current_style_meta['name']}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# ============================================================================
# STEP 3: Initialize MediaPipe Pose (Body tracking)
# ============================================================================
print("\n[3/6] Initializing MediaPipe pose detection...")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("   ✅ Body tracking ready")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
print("   ✅ Hand tracking ready")

# ============================================================================
# STEP 4: Initialize Camera
# ============================================================================
print("\n[4/6] Initializing webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("   ❌ Failed to open webcam!")
    exit(1)

print("   ✅ Camera opened")

# Warm up
time.sleep(2)
for i in range(10):
    cap.read()
    time.sleep(0.1)

print("   ✅ Camera ready")

# Get frame size
ret, test_frame = cap.read()
if ret:
    h, w = test_frame.shape[:2]
    print(f"   ✅ Frame size: {w}x{h}")
else:
    w, h = 640, 480

# ============================================================================
# STEP 5: Initialize Offscreen Renderer
# ============================================================================
print("\n[5/6] Creating offscreen 3D renderer...")
renderer = pyrender.OffscreenRenderer(w, h)
print(f"   ✅ Renderer ready: {w}x{h}")


def update_scene_lighting(frame, scene_obj, key_light, accent_light):
    """Match virtual lighting to the real-world frame for better realism."""
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.3), int(w * 0.7)
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return

    avg_bgr = cv2.mean(roi)[:3]
    avg_rgb = np.array(avg_bgr[::-1]) / 255.0
    avg_rgb = np.clip(avg_rgb, 0.05, 1.0)
    brightness = float(np.clip(avg_rgb.mean(), 0.05, 1.0))

    key_light.color = avg_rgb.tolist()
    key_light.intensity = float(np.interp(brightness, [0.05, 1.0], [1.8, 5.2]))

    rim_color = np.clip(avg_rgb + 0.2, 0.1, 1.0)
    accent_light.color = rim_color.tolist()
    accent_light.intensity = float(np.interp(1.0 - brightness, [0.0, 1.0], [1.2, 3.0]))

    ambient = (avg_rgb * 0.35 + 0.2).clip(0.1, 1.0)
    scene_obj.ambient_light = ambient.tolist()


# ============================================================================
# STEP 6: Main AR Loop
# ============================================================================
print("\n[6/6] Starting AR system...")
print("\n" + "="*60)
print("🧥 AR SYSTEM ACTIVE!")
print("="*60)
print("Controls:")
print("  q - Quit")
print("="*60 + "\n")

window_name = "Professional 3D Jacket AR"
cv2.namedWindow(window_name)

prev_time = time.time()

# Jacket node (will be added/removed each frame)
jacket_node = None
camera_node = None

scale_calibration = BASE_SCALE_CALIBRATION
height_offset_factor = BASE_HEIGHT_OFFSET
scale_step = 0.2
offset_step = 0.02

captures_dir = os.path.join(BASE_DIR, "captures")
os.makedirs(captures_dir, exist_ok=True)
feedback_text = ""
feedback_time = 0

logs_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(logs_dir, exist_ok=True)
session_log_path = os.path.join(
    logs_dir, f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
log_file = open(session_log_path, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["timestamp", "fps", "pose_confidence", "render_ms"])
last_log_time = time.time()
recording = False
video_writer = None
video_filename = ""
capture_fps = cap.get(cv2.CAP_PROP_FPS)
if capture_fps is None or capture_fps <= 1:
    capture_fps = 30.0
video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
last_gesture_time = 0.0


def compute_pose_confidence(landmark_list) -> float:
    """Average visibility for key torso/facial landmarks."""
    if not landmark_list:
        return 0.0
    key_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.MOUTH_LEFT.value,
        mp_pose.PoseLandmark.MOUTH_RIGHT.value,
    ]
    visibilities = [landmark_list[idx].visibility for idx in key_indices]
    return float(np.clip(np.mean(visibilities), 0.0, 1.0))


def refresh_style_metadata():
    """Sync cached metadata for the currently selected style."""
    global current_style_meta, current_style_name, current_style_file, current_style_description
    global current_color_variant_name, style_default_scale, style_default_offset
    current_style_meta = style_assets[current_style_idx % len(style_assets)]
    current_style_name = current_style_meta["name"]
    current_style_file = os.path.basename(current_style_meta["file"])
    current_style_description = current_style_meta.get("metadata", {}).get("description", "")
    variants = current_style_meta["color_variants"]
    variant_idx = current_style_meta["current_color_idx"] % len(variants)
    current_color_variant_name = variants[variant_idx]["name"]
    fit_defaults = current_style_meta.get("fit_defaults", {})
    style_default_scale = float(fit_defaults.get("scale", BASE_SCALE_CALIBRATION))
    style_default_offset = float(fit_defaults.get("offset", BASE_HEIGHT_OFFSET))


def apply_style_fit_defaults():
    """Reset live fit tuning to the current style's defaults."""
    global scale_calibration, height_offset_factor
    scale_calibration = style_default_scale
    height_offset_factor = style_default_offset


def cycle_color_variant():
    """Advance to the next color variant for the current style."""
    global pr_jacket
    style = style_assets[current_style_idx % len(style_assets)]
    style["current_color_idx"] = (style["current_color_idx"] + 1) % len(style["color_variants"])
    pr_jacket = build_jacket_mesh(current_style_idx)
    refresh_style_metadata()
    return current_color_variant_name


def detect_open_palm(hand_results):
    """Return True when a palm with five extended fingers is detected."""
    if not hand_results or not hand_results.multi_hand_landmarks:
        return False

    for hand_landmarks in hand_results.multi_hand_landmarks:
        landmarks = hand_landmarks.landmark
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
        bbox = coords.max(axis=0) - coords.min(axis=0)
        bbox_height = float(max(bbox[1], 1e-3))
        bbox_width = float(max(bbox[0], 1e-3))

        if bbox_height < 0.08:
            continue  # hand too small on screen to evaluate confidently

        fingers = [
            (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
            (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
        ]

        vertical_threshold = np.clip(0.25 * bbox_height, 0.015, 0.04)
        wrist_xy = coords[mp_hands.HandLandmark.WRIST.value, :2]
        extended = 0
        for tip_idx, pip_idx in fingers:
            tip = coords[tip_idx]
            pip = coords[pip_idx]
            if (pip[1] - tip[1]) > vertical_threshold:
                wrist_dist = np.linalg.norm(tip[:2] - wrist_xy)
                if wrist_dist > (0.35 * bbox_height):
                    extended += 1

        thumb_tip = coords[mp_hands.HandLandmark.THUMB_TIP.value]
        thumb_ip = coords[mp_hands.HandLandmark.THUMB_IP.value]
        horizontal_threshold = np.clip(0.30 * bbox_width, 0.02, 0.05)
        if abs(thumb_tip[0] - thumb_ip[0]) > horizontal_threshold:
            extended += 1

        if extended >= 5:
            return True

    return False


refresh_style_metadata()
apply_style_fit_defaults()


while True:
    ret, frame = cap.read()

    if not ret:
        time.sleep(0.05)
        continue

    # Mirror
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Match lighting to the real scene for better readability
    update_scene_lighting(frame, scene, main_light, rim_light)

    # Detect pose
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    pose_confidence = 0.0
    render_time_ms = 0.0
    palm_detected = False
    palm_triggered = False

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

        # Get 3D body landmarks
        landmarks = results.pose_landmarks.landmark

        # Key points with 3D coordinates
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # FACE landmarks - to avoid covering face!
        mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
        mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        # MATHEMATICAL SOLUTION: Calculate jacket size using perspective projection

        # Step 1: Convert normalized landmarks to PIXEL coordinates
        ls_px = np.array([ls.x * w, ls.y * h])
        rs_px = np.array([rs.x * w, rs.y * h])
        lh_px = np.array([lh.x * w, lh.y * h])
        rh_px = np.array([rh.x * w, rh.y * h])

        pose_confidence = compute_pose_confidence(landmarks)

        # Step 2: Calculate body dimensions in PIXELS (like the green box does)
        shoulder_width_pixels = np.linalg.norm(rs_px - ls_px)

        # Average shoulder and hip positions
        avg_shoulder_px = (ls_px + rs_px) / 2
        avg_hip_px = (lh_px + rh_px) / 2
        torso_height_pixels = np.linalg.norm(avg_shoulder_px - avg_hip_px)

        # Step 3: Use MOUTH position to position jacket below face!
        mouth_avg_px = np.array([
            (mouth_left.x + mouth_right.x) / 2 * w,
            (mouth_left.y + mouth_right.y) / 2 * h
        ])

        # Jacket should start BELOW mouth
        # Position jacket center = mouth_y + offset + (jacket_height / 2)
        jacket_top_offset = torso_height_pixels * height_offset_factor  # live-adjustable offset
        jacket_center_y = mouth_avg_px[1] + jacket_top_offset + (torso_height_pixels / 2)

        # X position: centered on shoulders
        center_px_x = (avg_shoulder_px[0] + avg_hip_px[0]) / 2

        # Convert center pixel coordinates to normalized (0-1)
        center_norm_x = center_px_x / w
        center_norm_y = jacket_center_y / h

        # Get average Z-depth from shoulders and hips
        avg_z = (ls.z + rs.z + lh.z + rh.z) / 4

        # Step 4: SIMPLE ADAPTIVE SCALING - Scale directly from pixel measurements

        # Base scale: How big is the body relative to frame height?
        pixel_scale = max(shoulder_width_pixels, torso_height_pixels) / h

        # Depth adjustment: MediaPipe's Z increases as you move away
        # When closer (smaller Z) = bigger depth_factor = bigger jacket
        # When farther (bigger Z) = smaller depth_factor = smaller jacket
        depth_factor = 1.0 / (1.0 + avg_z * 3.0)

        # Final scale with calibration constant
        # This makes the jacket match the green box size
        scale = pixel_scale * depth_factor * scale_calibration

        # Calculate camera depth for positioning
        z_depth = 2.0 - avg_z * 1.5

        # Create transformation matrix for jacket
        # 1. Scale
        transform = np.eye(4)
        transform[:3, :3] *= scale

        # 2. Translate jacket - positioned BELOW mouth to avoid covering face!
        # Convert normalized screen coords to camera space
        # X: left-right (0=left, 1=right) -> camera X axis
        # Y: top-bottom (0=top, 1=bottom) -> camera Y axis (INVERTED!)
        transform[0, 3] = (center_norm_x - 0.5) * 2.0      # X: center at 0.5
        transform[1, 3] = -(center_norm_y - 0.5) * 2.0     # Y: INVERTED, positioned below mouth
        transform[2, 3] = -z_depth                          # Z: depth (negative = away from camera)

        # Remove old jacket if exists
        if jacket_node is not None:
            scene.remove_node(jacket_node)
        if camera_node is not None:
            scene.remove_node(camera_node)

        # Add jacket to scene
        jacket_node = scene.add(pr_jacket, pose=transform)

        # Add camera
        camera_pose = np.eye(4)
        camera_node = scene.add(camera, pose=camera_pose)

        # Render jacket
        try:
            render_start = time.perf_counter()
            color, depth = renderer.render(scene)
            render_time_ms = (time.perf_counter() - render_start) * 1000.0

            # Create alpha mask from depth
            alpha = (depth > 0).astype(np.float32)

            # Expand alpha to 3 channels
            alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)

            # Blend jacket with frame
            jacket_rgb = color.astype(np.float32) / 255.0
            frame_float = frame.astype(np.float32) / 255.0

            blended = alpha_3ch * jacket_rgb + (1 - alpha_3ch) * frame_float
            frame = (blended * 255).astype(np.uint8)

            # Show confirmation
            cv2.putText(frame, "3D JACKET ACTIVE!", (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        except Exception as e:
            print(f"Render error: {e}")

    palm_detected = detect_open_palm(hand_results)
    if palm_detected:
        now = time.time()
        if (now - last_gesture_time) > 1.0:
            palm_triggered = True
            new_color_name = cycle_color_variant()
            feedback_text = f"Palm color swap → {new_color_name}"
            feedback_time = now
            last_gesture_time = now

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    if (current_time - last_log_time) >= 1.0:
        log_writer.writerow([
            datetime.now().isoformat(),
            f"{fps:.2f}",
            f"{pose_confidence:.3f}",
            f"{render_time_ms:.2f}"
        ])
        log_file.flush()
        last_log_time = current_time

    # UI
    cv2.putText(frame, "Professional 3D AR", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    overlay_lines = [
        f"Style: {current_style_name}",
        f"Color: {current_color_variant_name}",
        f"Asset: {current_style_file}",
        f"Scale x{scale_calibration:.1f} | Offset {height_offset_factor:.2f}",
        "Hotkeys: a/d scale  w/s height  c color  m style  p photo  v video  r reset"
    ]
    if current_style_description:
        overlay_lines.insert(3, current_style_description[:60])
    hud_base_y = max(150, h - 120)
    for idx, text in enumerate(overlay_lines):
        cv2.putText(frame, text, (20, hud_base_y + idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    palm_status = "triggered" if palm_triggered else ("detected" if palm_detected else "inactive")
    diag_lines = [
        f"Pose {pose_confidence*100:4.0f}% | Render {render_time_ms:5.1f} ms",
        f"Palm gesture: {palm_status}",
        f"Log: {os.path.basename(session_log_path)}"
    ]
    for idx, text in enumerate(diag_lines):
        cv2.putText(frame, text, (20, 120 + idx * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 241), 2)

    if recording:
        rec_text = f"REC ● {os.path.basename(video_filename) if video_filename else ''}"
        cv2.putText(frame, rec_text.strip(), (w - 260, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if feedback_text and (time.time() - feedback_time) < 2.5:
        cv2.putText(frame, feedback_text, (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (94, 241, 255), 2)

    if recording and video_writer is not None:
        video_writer.write(frame)

    cv2.imshow(window_name, frame)

    # Keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        scale_calibration = max(3.0, scale_calibration - scale_step)
    elif key == ord('d'):
        scale_calibration = min(9.0, scale_calibration + scale_step)
    elif key == ord('w'):
        height_offset_factor = max(0.3, height_offset_factor - offset_step)
    elif key == ord('s'):
        height_offset_factor = min(0.8, height_offset_factor + offset_step)
    elif key == ord('c'):
        new_color = cycle_color_variant()
        feedback_text = f"Color: {new_color}"
        feedback_time = time.time()
    elif key == ord('m'):
        current_style_idx = (current_style_idx + 1) % len(style_assets)
        pr_jacket = build_jacket_mesh(current_style_idx)
        refresh_style_metadata()
        apply_style_fit_defaults()
        feedback_text = f"Style: {current_style_name}"
        feedback_time = time.time()
    elif key == ord('r'):
        apply_style_fit_defaults()
        feedback_text = "Reset fit tuning"
        feedback_time = time.time()
    elif key == ord('p'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ar_jacket_{timestamp}.png"
        filepath = os.path.join(captures_dir, filename)
        cv2.imwrite(filepath, frame)
        feedback_text = f"Saved {filename}"
        feedback_time = time.time()
    elif key == ord('v'):
        if not recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(captures_dir, f"ar_jacket_{timestamp}.mp4")
            video_writer = cv2.VideoWriter(video_filename, video_fourcc, capture_fps, (w, h))
            if video_writer.isOpened():
                recording = True
                feedback_text = f"Recording {os.path.basename(video_filename)}"
            else:
                video_writer = None
                video_filename = ""
                feedback_text = "Video recorder unavailable"
            feedback_time = time.time()
        else:
            recording = False
            if video_writer is not None:
                video_writer.release()
            feedback_text = f"Saved {os.path.basename(video_filename)}"
            video_writer = None
            video_filename = ""
            feedback_time = time.time()

# Cleanup
cap.release()
if video_writer is not None:
    video_writer.release()
log_file.close()
cv2.destroyAllWindows()
pose.close()
hands.close()
renderer.delete()
print("\n✅ AR system closed")
