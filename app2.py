# -------------------------------------------------------------
# Streamlit MNIST demo — layout & behaviour like mlx.institute
# -------------------------------------------------------------
# Author : Han Wang  (revamped)
# Date   : 2025-05-02
# -------------------------------------------------------------


import streamlit as st
st.set_page_config(page_title="MNIST demo", page_icon="✏️", layout="wide")

import os, io, datetime, numpy as np, torch, torch.nn as nn, torchvision.transforms as T

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import altair as alt

# ─────────────────────────────────────────────────────────────
# 1. Model definition + load weights
# ─────────────────────────────────────────────────────────────
def cnn_model():
    return nn.Sequential(
        nn.ZeroPad2d(2),
        nn.Conv2d(1, 16, 5, 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.LazyLinear(10),
        nn.Softmax(dim=1)
    )

@st.cache_resource
def load_model():
    m = cnn_model()
    ckpt = "./checkpoint/250501_mnist_cnn_pt1_cpu.pt"
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m

model = load_model()

# ─────────────────────────────────────────────────────────────
# 2. Helper: canvas RGBA  → centred 28×28 tensor
# ─────────────────────────────────────────────────────────────
def preprocess_rgba(arr: np.ndarray) -> torch.Tensor:
    """
    280×280 RGBA → 1×28×28 float tensor in [0,1], centred like MNIST.
    """
    img = Image.fromarray(arr.astype("uint8"), mode="RGBA").convert("L")   # to gray
    img = ImageOps.invert(img)                                             # white-stroke
    img = img.point(lambda x: 0 if x < 30 else 255, "L")                   # binarise
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = img.resize((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(img, (4, 4))                                              # centre
    return T.ToTensor()(canvas)                                            # (1,28,28)

# ─────────────────────────────────────────────────────────────
# 3. UI — title & layout
# ─────────────────────────────────────────────────────────────


left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("## Draw a digit")
    canvas = st_canvas(
        fill_color="#000000",
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        stroke_width=12,
        height=280, width=280,
        key="canvas",
    )
    col_clear, _ = st.columns([1,4])
    if col_clear.button("🗑️ Clear"):
        st.experimental_rerun()

# ─────────────────────────────────────────────────────────────
# 4. If canvas has ink → run inference
# ─────────────────────────────────────────────────────────────
if canvas.image_data is not None and np.any(canvas.image_data[..., :3] != 0):
    X = preprocess_rgba(canvas.image_data).unsqueeze(0)   # (1,1,28,28)
    with torch.no_grad():
        probs = model(X)[0].numpy()
    pred_digit = int(np.argmax(probs))
    topk_idx   = np.argsort(probs)[-3:][::-1]
    topk_prob  = probs[topk_idx]

    with right:
        st.markdown("## Result")
        st.metric(label="Prediction", value=f"{pred_digit}", delta=f"{probs[pred_digit]*100:.1f}%")

        # preview the 28×28 tensor the model saw
        st.markdown("##### Model input")
        st.image(X.squeeze(0).squeeze(0).numpy(), clamp=True, width=120)

        # Top-3 bar chart (Altair)
        chart_df = (
            alt.Chart(
                data={"index": topk_idx.astype(str), "prob": topk_prob}
            )
            .mark_bar()
            .encode(
                x=alt.X("index:N", title="Digit"),
                y=alt.Y("prob:Q", title="Confidence", scale=alt.Scale(domain=[0,1])),
                tooltip=["prob"]
            )
        )
        st.altair_chart(chart_df, use_container_width=True)

        st.caption("Top-3 classes")
else:
    with right:
        st.markdown("## Result")
        st.info("Draw a digit to get a prediction.")

# ─────────────────────────────────────────────────────────────
# 5. Footer
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:2rem;margin-bottom:0">
    <small>Model: 2-conv CNN trained by Han Wang · Front-end rebuilt to match
    <a href="https://mnist-example.mlx.institute/" target="_blank">mlx institute demo</a>.</small>
    """,
    unsafe_allow_html=True,
)