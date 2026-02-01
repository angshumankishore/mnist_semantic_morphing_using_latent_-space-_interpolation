import streamlit as st
import torch
import cv2
import numpy as np
import os
from vae_model import VAE
from image_utils import load_user_image, load_images_from_file_list

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
Two input modes supported:

- Text numbers: type multi-digit numbers (e.g. `12`) where each character is a digit image.
- Upload images: upload one image per side (each image may contain multiple digits side-by-side) or multiple single-digit images. The app will try to split combined images automatically.
""")

mode = st.radio("Input mode", ["Text numbers", "Upload images"])

src_digits = []
tgt_digits = []

if mode == "Text numbers":
    src = st.text_input("Source Number (e.g. 12)")
    tgt = st.text_input("Target Number (e.g. 34)")

    if st.button("Generate Morphing Video"):
        if not is_valid_number_string(src) or not is_valid_number_string(tgt):
            st.error("❌ Invalid input. Use multi-digit, non-repeating numbers only.")
            st.stop()

        if len(src) != len(tgt):
            st.error("❌ Source and target must have the same length.")
            st.stop()

        for s, t in zip(src, tgt):
            # load_user_image when given a single-character digit returns a list
            # with one tensor; extract the first element
            imgs1 = load_user_image(s)
            imgs2 = load_user_image(t)
            src_digits.append(imgs1[0])
            tgt_digits.append(imgs2[0])

        video_segments = []
        with st.spinner("Generating morphs..."):
            for i, (img1, img2) in enumerate(zip(src_digits, tgt_digits)):
                temp_video = os.path.join(TEMP_DIR, f"segment_{i}.avi")
                morph_digits(img1, img2, temp_video)
                video_segments.append(temp_video)

            final_video = os.path.join(VIDEO_DIR, "final_morph.avi")
            stitch_videos(video_segments, final_video)

        st.success("✅ Video generated!")
        st.video(final_video)

else:
    st.markdown("Upload one or more files for Source and Target. If you upload a single image containing multiple digits (e.g. '12'), the app will try to split it into individual digits.")
    src_files = st.file_uploader("Source image(s) — upload one image containing multiple digits or multiple single-digit images", accept_multiple_files=True, type=["png","jpg","jpeg"] )
    tgt_files = st.file_uploader("Target image(s)", accept_multiple_files=True, type=["png","jpg","jpeg"] )

    if st.button("Generate Morphing Video"):
        if not src_files or not tgt_files:
            st.error("❌ Please upload source and target images.")
            st.stop()

        # flatten files into list of tensors (keeps order)
        try:
            src_digits = load_images_from_file_list(src_files)
            tgt_digits = load_images_from_file_list(tgt_files)
        except Exception as e:
            st.error(f"Error loading images: {e}")
            st.stop()

        if len(src_digits) < 2 or len(tgt_digits) < 2:
            st.error("❌ Need at least two digits per side (multi-digit numbers).")
            st.stop()

        if len(src_digits) != len(tgt_digits):
            st.error("❌ Source and target must have the same number of digits.")
            st.stop()

        video_segments = []
        with st.spinner("Generating morphs..."):
            for i, (img1, img2) in enumerate(zip(src_digits, tgt_digits)):
                temp_video = os.path.join(TEMP_DIR, f"segment_{i}.avi")
                morph_digits(img1, img2, temp_video)
                video_segments.append(temp_video)

            final_video = os.path.join(VIDEO_DIR, "final_morph.avi")
            stitch_videos(video_segments, final_video)

        st.success("✅ Video generated!")
        st.video(final_video)
