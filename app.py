import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
import cv2
import numpy as np
import os
import tempfile
from PIL import Image

from model import (
    FullDeepFakeDetector,
    compute_fft_batch,
)

# -----------------------------------------------------------
# Streamlit Page Settings
# -----------------------------------------------------------
st.set_page_config(page_title="MarkV Deepfake Detector", layout="wide")

# -----------------------------------------------------------
# Load Models (Cached)
# -----------------------------------------------------------
@st.cache_resource
def load_models():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "MarkV_weights.pth"

    if not os.path.exists(model_path):
        st.error("Missing MarkV_weights.pth file.")
        return None, None, None, None

    model = FullDeepFakeDetector(base_weights_path=None, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    mtcnn = MTCNN(image_size=224, device=device, post_process=False)

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return model, mtcnn, transform, device


model, mtcnn, tf, device = load_models()

# -----------------------------------------------------------
# Video Face Extraction
# -----------------------------------------------------------
def process_video_for_analysis(video_file, mtcnn, transform, device, frame_skip=10):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
        vid_path = tfile.name

    cap = cv2.VideoCapture(vid_path)
    frames, crops, frame_ids = [], [], []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)

            if boxes is not None:
                faces = mtcnn.extract(rgb, boxes, save_path=None)

                if faces is not None:

                    # Store original crop (for display)
                    face_pil = transforms.ToPILImage()(faces[0].byte())
                    crops.append(face_pil)

                    # Prepare NN input
                    face = faces[0] / 255.0
                    if face.ndim == 2:
                        face = face.unsqueeze(0).repeat(3, 1, 1)
                    elif face.shape[0] == 1:
                        face = face.repeat(3, 1, 1)

                    face = transform(face).to(device)
                    frames.append(face)
                    frame_ids.append(idx)

        idx += 1

    cap.release()
    os.remove(vid_path)

    return frames, crops, frame_ids

# -----------------------------------------------------------
# Frame-Level Prediction + Feature Extraction
# -----------------------------------------------------------
def get_frame_scores_and_features(frame_model, frames):
    """
    For each frame tensor:
      - run frame model + temporal head (single-step) to get a real-score
      - collect the 768-dim feature vector after cross-attention
    """
    scores_real = []
    features = []

    with torch.no_grad():
        for face in frames:
            batch = face.unsqueeze(0)  # (1, 3, H, W)
            freq  = compute_fft_batch(batch)

            # 768-dim feature vector (spatial+freq after cross-attention)
            feat  = frame_model(batch, freq)          # (1, 768)
            features.append(feat.squeeze(0).cpu())

            # Per-frame score via temporal head with a single timestep
            seq   = feat.unsqueeze(1)                 # (1, 1, 768)
            logit = model.temporal(seq)               # (1, 1)
            score_real = torch.sigmoid(logit).item()
            scores_real.append(score_real)

    if features:
        features = torch.stack(features)              # (N, 768)
    else:
        features = torch.empty(0, 768)

    return scores_real, features

# -----------------------------------------------------------
# Prototype Computation for Feature-Delta XAI
# -----------------------------------------------------------
def compute_prototypes(features, real_scores, top_k=5):
    """
    Compute prototype embeddings for "real-like" and "fake-like" frames
    using the current video's frames only.
    """
    if features.numel() == 0:
        return None, None

    scores_real = torch.tensor(real_scores, dtype=torch.float32)
    scores_fake = 1.0 - scores_real

    k = min(top_k, features.shape[0])

    real_top_idx = torch.topk(scores_real, k=k).indices
    fake_top_idx = torch.topk(scores_fake, k=k).indices

    proto_real = features[real_top_idx].mean(dim=0, keepdim=True)  # (1, 768)
    proto_fake = features[fake_top_idx].mean(dim=0, keepdim=True)  # (1, 768)

    return proto_real, proto_fake

# -----------------------------------------------------------
# Temporal Influence from per-frame confidence
# -----------------------------------------------------------
def compute_temporal_influence(real_scores):
    """
    Convert how confident each frame is (distance from 0.5) into
    a normalized "influence" score that sums to 1 across frames.
    """
    scores_real = np.array(real_scores, dtype=np.float32)
    influence_raw = np.abs(scores_real - 0.5)        # frames far from 0.5 are more influential
    total = influence_raw.sum() + 1e-8
    influence = influence_raw / total
    return influence  # numpy array of shape (N,)

# -----------------------------------------------------------
# Model-based XAI for a single frame (no Grad-CAM)
# -----------------------------------------------------------
def analyze_model_features_for_frame(model, face_tensor):
    """
    Runs the spatial + frequency branches manually to get:
      - energy in low/mid/high frequency bands
      - relative importance of spatial vs frequency features
    """
    with torch.no_grad():
        x = face_tensor.unsqueeze(0)  # (1, 3, H, W)
        freq_in = compute_fft_batch(x)

        fm = model.frame_model

        # ---- Spatial branch ----
        s = fm.spatial(x)
        s = fm.spatial_attention(s)
        s = fm.spatial_pool(s).flatten(1)
        s_vec = fm.spatial_fc(s)  # (1, 512)

        # ---- Frequency branch (multi-scale) ----
        fb = fm.freq_branch
        low = fb.low_freq(freq_in)
        mid = fb.mid_freq(freq_in)
        high = fb.high_freq(freq_in)

        # Raw energy in each frequency band
        low_energy  = low.abs().mean().item()
        mid_energy  = mid.abs().mean().item()
        high_energy = high.abs().mean().item()

        # Fuse as in forward()
        h, w = mid.shape[2], mid.shape[3]
        low_r  = F.interpolate(low,  size=(h, w), mode='bilinear', align_corners=False)
        high_r = F.interpolate(high, size=(h, w), mode='bilinear', align_corners=False)
        multi  = torch.cat([low_r, mid, high_r], dim=1)
        fused  = fb.fusion(multi).flatten(1)
        f_vec  = fb.fc(fused)  # (1, 256)

        # ---- Cross-attention importance ----
        combined = torch.cat([s_vec, f_vec], dim=1)  # (1, 768)
        att = fm.cross_attention(combined)           # (1, 768)

        spatial_imp = att[:, :s_vec.shape[1]].mean().item()
        freq_imp    = att[:, s_vec.shape[1]:].mean().item()

    return {
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "spatial_imp": spatial_imp,
        "freq_imp": freq_imp,
    }

# -----------------------------------------------------------
# Natural-language Explanation Builder
# -----------------------------------------------------------
def build_explanation_for_frame(
    is_video_fake,
    real_score,
    sim_real,
    sim_fake,
    freq_info,
    spatial_imp,
    freq_imp,
    temporal_contrib
):
    """
    Turn numeric XAI metrics into human-readable text.
    """
    fake_score = 1.0 - real_score
    lines = []

    # Overall confidence context
    if fake_score > 0.8:
        lines.append("The model is highly confident this frame is manipulated.")
    elif fake_score > 0.6:
        lines.append("The model sees noticeable signs of manipulation in this frame.")
    elif fake_score > 0.55:
        lines.append("This frame shows mild signals that lean towards manipulation.")
    elif real_score > 0.8:
        lines.append("The model is highly confident this frame looks authentic.")
    elif real_score > 0.6:
        lines.append("This frame appears mostly consistent with real facial patterns.")
    else:
        lines.append("This frame lies near the model's decision boundary between real and fake.")

    # Feature-cluster similarity (prototype-based)
    if sim_real is not None and sim_fake is not None:
        if is_video_fake:
            delta = sim_fake - sim_real
            if delta > 0.15:
                lines.append(f"The frame's feature embedding is much closer to the model's fake cluster than the real cluster (Δcos ≈ {delta:.2f}).")
            elif delta > 0.05:
                lines.append(f"The feature embedding is slightly biased towards the fake cluster (Δcos ≈ {delta:.2f}).")
            else:
                lines.append("The embedding sits between typical real and fake patterns for this video.")
        else:
            delta = sim_real - sim_fake
            if delta > 0.15:
                lines.append(f"The frame's feature embedding is strongly aligned with the real-face prototype (Δcos ≈ {delta:.2f} in favor of real).")
            elif delta > 0.05:
                lines.append(f"The features lean somewhat towards the real-face prototype (Δcos ≈ {delta:.2f}).")
            else:
                lines.append("The embedding is balanced between real-like and fake-like clusters for this video.")

    # Spatial vs frequency importance
    total_imp = spatial_imp + freq_imp + 1e-8
    spatial_pct = 100.0 * spatial_imp / total_imp
    freq_pct = 100.0 * freq_imp / total_imp

    if freq_pct > 60:
        lines.append(f"The decision is driven more by frequency-domain artifacts (~{freq_pct:.1f}% importance) than by facial shape and texture.")
    elif spatial_pct > 60:
        lines.append(f"The decision relies mostly on spatial facial details (~{spatial_pct:.1f}% importance) such as structure and texture.")
    else:
        lines.append(f"The model uses a balanced mix of spatial (~{spatial_pct:.1f}%) and frequency (~{freq_pct:.1f}%) cues.")

    # Frequency band contributions
    low_p, mid_p, high_p = freq_info
    if high_p >= max(low_p, mid_p) and high_p > 40:
        lines.append(f"High-frequency components are unusually strong (~{high_p:.1f}% of frequency energy), which often indicates GAN sharpening or compression artifacts.")
    elif mid_p >= max(low_p, high_p) and mid_p > 40:
        lines.append(f"Mid-frequency texture patterns dominate (~{mid_p:.1f}% of frequency energy), suggesting blending or warping in facial regions.")
    elif low_p > 45:
        lines.append(f"Low-frequency energy is dominant (~{low_p:.1f}%), meaning the model focuses more on overall shading and smoothness.")
    else:
        lines.append(f"Frequency energy is fairly evenly distributed across low ({low_p:.1f}%), mid ({mid_p:.1f}%), and high ({high_p:.1f}%) bands.")

    # Temporal contribution
    if temporal_contrib > 0.25:
        lines.append(f"This frame is one of the most influential frames in the clip, contributing roughly {temporal_contrib*100:.1f}% of the model's per-frame influence.")
    elif temporal_contrib > 0.15:
        lines.append(f"This frame has a moderate impact on the overall video decision (~{temporal_contrib*100:.1f}% influence).")
    else:
        lines.append(f"This frame plays a smaller role in the final decision (~{temporal_contrib*100:.1f}% influence).")

    return " ".join(lines)

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.title("MarkV Deepfake Detector 🔍")

if model is None:
    st.stop()

uploaded = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])

if uploaded:

    st.video(uploaded.getvalue())

    if st.button("Analyze Video"):

        # Step 1 — Face detection
        with st.spinner("Step 1/3 — Detecting faces..."):
            frames, crops, frame_ids = process_video_for_analysis(uploaded, mtcnn, tf, device)

        if not frames:
            st.error("No faces detected in the video.")
            st.stop()

        # Step 2 — Temporal deepfake detection (video-level)
        with st.spinner("Step 2/3 — Running temporal model..."):
            seq = frames.copy()
            if len(seq) < 10:
                seq += [seq[-1]] * (10 - len(seq))

            # Use up to 10 frames for the video-level decision
            seq_tensor = torch.stack(seq[:10]).unsqueeze(0)  # (1, T, 3, 224, 224)
            with torch.no_grad():
                video_logit = model(seq_tensor).item()
                real_prob = torch.sigmoid(torch.tensor(video_logit)).item()
            fake_prob = 1.0 - real_prob
            is_fake = fake_prob > 0.5

        # Step 3 — Frame-by-frame analysis (per-frame scores + embeddings)
        with st.spinner("Step 3/3 — Analyzing frames..."):
            real_scores, frame_features = get_frame_scores_and_features(model.frame_model, frames)
            fake_scores = [1.0 - rs for rs in real_scores]

            # Prototypes for feature-delta XAI
            proto_real, proto_fake = compute_prototypes(frame_features, real_scores)

            # Temporal influence based on per-frame confidence
            temporal_influence = compute_temporal_influence(real_scores)

        # Final Video Verdict
        if is_fake:
            st.error(f"FAKE detected — Confidence: {fake_prob:.1%}")
        else:
            st.success(f"REAL video — Confidence: {real_prob:.1%}")

        st.divider()

        # XAI Section
        with st.expander("Frame-by-Frame Analysis (XAI)"):

            # Select top frames
            if is_fake:
                top_ids = np.argsort(fake_scores)[-5:]
            else:
                top_ids = np.argsort(real_scores)[-5:]

            top_ids = sorted(top_ids.tolist())

            cols = st.columns(len(top_ids))

            for col_idx, frame_idx in enumerate(top_ids):
                with cols[col_idx]:

                    st.caption(f"Frame #{frame_ids[frame_idx]}")

                    face_pil = crops[frame_idx]
                    st.image(face_pil, use_container_width=True)

                    # Score
                    frame_real_score = real_scores[frame_idx]
                    st.write(f"Real Score: **{frame_real_score*100:.1f}%**")

                    # Cosine similarity to prototypes
                    sim_real = sim_fake = None
                    if proto_real is not None and proto_fake is not None and frame_features.numel() > 0:
                        feat = frame_features[frame_idx].unsqueeze(0)  # (1, 768)
                        sim_real = F.cosine_similarity(feat, proto_real, dim=1).item()
                        sim_fake = F.cosine_similarity(feat, proto_fake, dim=1).item()

                    # Model-based feature analysis (freq bands + spatial/freq importance)
                    frame_tensor = frames[frame_idx]
                    feat_info = analyze_model_features_for_frame(model, frame_tensor)

                    low_e = feat_info["low_energy"]
                    mid_e = feat_info["mid_energy"]
                    high_e = feat_info["high_energy"]
                    total_e = low_e + mid_e + high_e + 1e-8
                    low_p = 100.0 * low_e / total_e
                    mid_p = 100.0 * mid_e / total_e
                    high_p = 100.0 * high_e / total_e

                    spatial_imp = feat_info["spatial_imp"]
                    freq_imp = feat_info["freq_imp"]

                    # Temporal contribution for this frame
                    t_contrib = float(temporal_influence[frame_idx])

                    # Generate explanation text
                    explanation = build_explanation_for_frame(
                        is_video_fake=is_fake,
                        real_score=frame_real_score,
                        sim_real=sim_real,
                        sim_fake=sim_fake,
                        freq_info=(low_p, mid_p, high_p),
                        spatial_imp=spatial_imp,
                        freq_imp=freq_imp,
                        temporal_contrib=t_contrib,
                    )

                    st.write("**Explanation:**")
                    st.write(explanation)
