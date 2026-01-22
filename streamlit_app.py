import streamlit as st
import torch
import cv2
import numpy as np
import os
from vae_model import VAE, load_user_image

# =============================
# CONFIG
# =============================
DEVICE = torch.device("cpu")
MODEL_PATH = "vae_mnist.pth"
VIDEO_DIR = "videos"
TEMP_DIR = "videos/temp"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

STEPS = 40
FRAME_SIZE = (256, 256)
FPS = 15

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    model = VAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

vae = load_model()

# =============================
# HELPERS
# =============================
def is_valid_number_string(s):
    if not s.isdigit():
        return False
    if len(s) < 2:
        return False
    if len(set(s)) == 1:
        return False
    return True


def morph_digits(img1, img2, out_path):
    frames = []

    with torch.no_grad():
        z1, _ = vae.encode(img1)
        z2, _ = vae.encode(img2)

        for alpha in np.linspace(0, 1, STEPS):
            z = (1 - alpha) * z1 + alpha * z2
            decoded = vae.decode(z).view(28, 28).cpu().numpy()
            frame = (decoded * 255).astype(np.uint8)
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(frame)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(out_path, fourcc, FPS, FRAME_SIZE)

    for f in frames:
        video.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))

    video.release()


def stitch_videos(video_paths, final_path):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(final_path, fourcc, FPS, FRAME_SIZE)

    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()

# =============================
# STREAMLIT UI
# =============================
st.title("🔢 MNIST Latent Space Morphing (VAE)")

st.markdown("""
Enter **multi-digit numbers only**  
Examples: `19 → 33`, `248 → 572`
""")

src = st.text_input("Source Number")
tgt = st.text_input("Target Number")

if st.button("Generate Morphing Video"):
    if not is_valid_number_string(src) or not is_valid_number_string(tgt):
        st.error("❌ Invalid input. Use multi-digit, non-repeating numbers only.")
        st.stop()

    if len(src) != len(tgt):
        st.error("❌ Source and target must have the same length.")
        st.stop()

    video_segments = []

    with st.spinner("Generating morphs..."):
        for i, (s, t) in enumerate(zip(src, tgt)):
            img1 = load_user_image(s)
            img2 = load_user_image(t)

            temp_video = os.path.join(TEMP_DIR, f"segment_{i}.avi")
            morph_digits(img1, img2, temp_video)
            video_segments.append(temp_video)

        final_video = os.path.join(VIDEO_DIR, "final_morph.avi")
        stitch_videos(video_segments, final_video)

    st.success("✅ Video generated!")
    st.video(final_video)
