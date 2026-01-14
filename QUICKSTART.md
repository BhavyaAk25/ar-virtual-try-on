# Quick Start Guide - 3D AR Jacket Try-On

Get your AR jacket system running in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip3 install opencv-python mediapipe numpy trimesh pyrender
```

## Step 2: Run the Application

```bash
python3 jacket_ar.py
```

## Step 3: Try It Out!

- Stand 2-3 feet from your webcam
- Keep your shoulders and hips visible
- The blue 3D jacket will appear on your torso
- Move closer/farther and watch it adapt!
- Press 'q' to quit

## That's it! 🎉

Your face will be visible (jacket stays below your mouth), and the jacket will automatically scale to fit your body size.

---

## Quick Troubleshooting

**Jacket not visible?**
- Make sure shoulders and hips are in frame
- Check that `jacket.glb` file exists
- Ensure good lighting

**Jacket too big/small?**
- Edit `jacket_ar.py` line 234: change `6.5` to adjust size
- Move closer or farther from camera

**Jacket covering your face?**
- Edit `jacket_ar.py` line 209: increase `0.5` to `0.6` or higher

**Need detailed help?**
- See the full [README.md](README.md) for complete documentation
- Check the "What Worked / What Didn't Work" section for technical details
