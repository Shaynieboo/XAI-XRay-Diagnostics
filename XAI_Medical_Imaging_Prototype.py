"""
XAI-Enhanced X-Ray Diagnostics — Dark Mode Dashboard
======================================================
Streamlit application for bone fracture detection with Grad-CAM explainability.
Author : Shayne Felicien-Brown (w1990606)
Degree : BSc Computer Science FT — University of Westminster
"""

import io
import os
import re
import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER


# PAGE CONFIG

st.set_page_config(
    page_title="XAI X-Ray Diagnostics",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
}
.stApp { background: #0a0e1a !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1628 !important;
    border-right: 1px solid #1e2d4a !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] .stFileUploader > div {
    border: 1px dashed #2563eb !important;
    background: #111827 !important;
    border-radius: 10px !important;
}

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0f1628 0%, #162040 50%, #0f1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #2563eb, #06b6d4, #8b5cf6);
}
/* CHANGED: dash-title font-size from 1.75rem to 2.2rem */
.dash-title {
    font-size: 2.2rem; font-weight: 700; color: #f1f5f9;
    margin: 0; letter-spacing: -0.5px;
}
/* CHANGED: dash-sub font-size from 0.78rem to 0.95rem, color from #64748b to #94a3b8 */
.dash-sub {
    color: #94a3b8; font-size: 0.95rem; margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.status-pill {
    display: flex; align-items: center; gap: 8px;
    background: #0d1f3c; border: 1px solid #1e3a5f;
    border-radius: 999px; padding: 8px 18px;
    /* CHANGED: font-size from 0.75rem to 0.9rem */
    font-size: 0.9rem; font-family: 'IBM Plex Mono', monospace; color: #94a3b8;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 6px #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Prediction badges */
.badge-anomaly {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444; border-radius: 10px;
    padding: 18px 22px; text-align: center; margin: 6px 0;
}
.badge-normal {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e; border-radius: 10px;
    padding: 18px 22px; text-align: center; margin: 6px 0;
}
.badge-label { font-size: 1.5rem; font-weight: 700; }
.badge-sub { font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; opacity: 0.7; margin-top: 2px; }

/* Confidence */
.conf-track {
    background: #1e2d4a; border-radius: 99px; height: 8px;
    margin: 6px 0 14px 0; overflow: hidden;
}
.conf-fill-anom { background: linear-gradient(90deg, #dc2626, #f97316); height: 100%; border-radius: 99px; }
.conf-fill-norm { background: linear-gradient(90deg, #16a34a, #22c55e); height: 100%; border-radius: 99px; }

/* XAI */
.xai-summary {
    background: #0d1f3c; border-left: 3px solid #2563eb;
    border-radius: 0 10px 10px 0; padding: 16px 20px; margin: 10px 0;
}
.xai-summary-warn {
    background: #1a0d00; border-left: 3px solid #f97316;
    border-radius: 0 10px 10px 0; padding: 16px 20px; margin: 10px 0;
}
.xai-section {
    background: #111827; border: 1px solid #1e2d4a;
    border-radius: 10px; padding: 16px 18px; margin: 8px 0;
}
.xai-section-title {
    font-size: 0.65rem; font-family: 'IBM Plex Mono', monospace;
    font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;
}
.xai-text { font-size: 0.88rem; line-height: 1.65; color: #cbd5e1; }

/* Region bars */
.region-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
/* CHANGED: region-name font-size from 0.72rem to 0.85rem, color from #94a3b8 to #cbd5e1 */
.region-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #cbd5e1; min-width: 145px; }
.region-track { flex: 1; background: #1e2d4a; border-radius: 99px; height: 6px; overflow: hidden; }
/* CHANGED: region-val font-size from 0.7rem to 0.82rem, color from #64748b to #94a3b8 */
.region-val { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #94a3b8; min-width: 38px; text-align: right; }

/* Metrics */
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0; }
.metric-box { background: #111827; border: 1px solid #1e2d4a; border-radius: 10px; padding: 14px 16px; text-align: center; }
.metric-val { font-size: 1.25rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; color: #60a5fa; }
.metric-lbl { font-size: 0.65rem; letter-spacing: 1px; text-transform: uppercase; color: #475569; margin-top: 3px; }

/* Step cards */
.step-card { background: #0f1628; border: 1px solid #1e2d4a; border-radius: 12px; padding: 36px 28px; text-align: center; }
.step-icon { font-size: 3rem; margin-bottom: 12px; }
.step-title { font-weight: 600; font-size: 1.25rem; color: #f1f5f9; margin: 8px 0; }
.step-desc { font-size: 1rem; color: #94a3b8; line-height: 1.65; }

/* Disclaimer */
.disclaimer { background: #1a1400; border: 1px solid #78350f; border-radius: 8px; padding: 10px 14px; font-size: 0.78rem; color: #fbbf24; margin-top: 12px; }

/* Labels */
.card-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #475569; font-family: 'IBM Plex Mono', monospace; margin-bottom: 10px; }

/* Streamlit overrides */
.stDownloadButton > button {
    background: #111827 !important; color: #60a5fa !important;
    border: 1px solid #1e3a5f !important; border-radius: 8px !important; font-weight: 600 !important;
}
div[data-testid="stExpander"] { background: #0f1628 !important; border: 1px solid #1e2d4a !important; border-radius: 10px !important; }
div[data-testid="stExpander"] summary { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# CONSTANTS

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Anomaly", "Normal"]
MODEL_PATH  = "models/fracture_resnet18_best.pth"


# MODEL

@st.cache_resource
def load_model():
    mdl = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    mdl.fc = nn.Linear(mdl.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        mdl.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    mdl.eval().to(DEVICE)
    return mdl

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess(img: Image.Image) -> torch.Tensor:
    return _transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)


# GRAD-CAM

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = self.gradients = None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):    self.activations = out.detach()
    def _bwd(self, _, __, gout):   self.gradients   = gout[0].detach()

    def generate(self, x: torch.Tensor, cls: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        logits[:, cls].sum().backward(retain_graph=True)
        w   = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def make_overlay(img: Image.Image, heatmap: np.ndarray, alpha=0.45) -> np.ndarray:
    base = cv2.resize(np.array(img.convert("RGB")), (224, 224))
    jet  = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 1 - alpha, jet, alpha, 0)


# XAI ENGINE

def analyse_regions(heatmap: np.ndarray) -> dict:
    h, w = heatmap.shape
    zones = {
        "Proximal (top)"  : heatmap[:h//3, :],
        "Shaft (middle)"  : heatmap[h//3: 2*h//3, :],
        "Distal (bottom)" : heatmap[2*h//3:, :],
        "Medial (left)"   : heatmap[:, :w//2],
        "Lateral (right)" : heatmap[:, w//2:],
    }
    acts = {n: float(z.mean()) for n, z in zones.items()}
    peak = max(acts, key=acts.get)
    return dict(acts=acts, peak=peak, peak_val=acts[peak],
                high_ratio=float((heatmap > 0.5).sum()) / heatmap.size)


def build_explanation(label: str, conf: float, rd: dict) -> dict:
    peak, pv, ratio = rd["peak"], rd["peak_val"], rd["high_ratio"]

    if ratio < 0.10:
        spread_tag  = "highly localised"
        spread_desc = "The model's attention is tightly focused on a small area — typical of a concentrated structural irregularity such as a cortical break."
    elif ratio < 0.25:
        spread_tag  = "moderately focused"
        spread_desc = "Activation covers a defined region with some surrounding context — consistent with a fracture zone and local response."
    else:
        spread_tag  = "broadly distributed"
        spread_desc = "Activation is spread across a large area, which may indicate diffuse bone changes or genuine model uncertainty."

    if conf >= 0.90:
        conf_interp = f"At {conf:.1%}, this is a high-confidence prediction. The model detected clear, visually distinct features consistent with its training distribution."
    elif conf >= 0.75:
        conf_interp = f"At {conf:.1%}, confidence is moderate. Features are present but subtle — specialist review is advisable."
    else:
        conf_interp = f"At {conf:.1%}, confidence is relatively low. This case should be reviewed by a radiologist regardless of the AI result."

    if label == "Anomaly":
        simple = f"The model identified a likely bone anomaly with **{conf:.1%} confidence**. The Grad-CAM heatmap highlights the **{peak.lower()}** region as the primary area of concern."

        if "proximal" in peak.lower():
            region_detail = (f"Strongest activation ({pv:.2f}) was in the proximal region. "
                "Proximal patterns are linked to periarticular fractures, growth-plate injuries, "
                "or joint-adjacent cortical disruptions — structures under high compressive stress.")
            clinical = ("Periarticular fractures are often underdiagnosed due to overlapping joint shadows. "
                "Dedicated oblique views or cross-sectional imaging (CT/MRI) may be warranted.")
        elif "shaft" in peak.lower():
            region_detail = (f"Strongest activation ({pv:.2f}) was in the mid-shaft region. "
                "Mid-shaft anomalies are classically associated with diaphyseal fractures, cortical "
                "thinning, or periosteal reactions — frequently missed in resource-limited settings.")
            clinical = ("Diaphyseal fractures can present as hairline cracks invisible on a single "
                "radiographic projection. Consider orthogonal views if clinical suspicion is high.")
        elif "distal" in peak.lower():
            region_detail = (f"Strongest activation ({pv:.2f}) was at the distal end. "
                "Distal activation is consistent with metaphyseal fractures, avulsion injuries, "
                "or impaction fractures near growth plates.")
            clinical = ("Distal fractures near growth plates are significant in younger patients. "
                "Salter-Harris classification and CT evaluation may be required if the film is inconclusive.")
        else:
            region_detail = (f"Strongest activation ({pv:.2f}) was along the {peak.lower()} cortical margin. "
                "Lateral/medial activation can indicate longitudinal cortical irregularity or stress reactions.")
            clinical = ("Cortical margin irregularities may represent stress fractures or subperiosteal haematomas. "
                "Clinical history should guide further workup.")

        pattern = (f"The heatmap activation is **{spread_tag}**. {spread_desc} "
            "In the overlay image, red/warm colours show the most influential regions; blue areas had minimal influence.")

    else:
        simple = f"No significant bone anomaly detected. The model predicted **Normal** with **{conf:.1%} confidence**."
        region_detail = (f"Peak activation ({pv:.2f}) in the {peak.lower()} region is expected for a Normal prediction. "
            "The model evaluates the whole bone structure and uses high-gradient areas to confirm the "
            "absence of structural disruption, not to flag pathology.")
        clinical = ("A Normal prediction means no fracture-consistent pattern was detected. "
            "However, this is a research prototype — subtle or non-displaced fractures can fall "
            "below the detection threshold. Clinical correlation is always recommended.")
        pattern = (f"Activation is **{spread_tag}**. {spread_desc} "
            "Diffuse or low-intensity heatmaps are expected for normal X-rays.")

    return dict(
        simple        = simple,
        region_detail = region_detail,
        pattern       = pattern,
        conf_interp   = conf_interp,
        clinical      = clinical,
        regions       = sorted(rd["acts"].items(), key=lambda x: x[1], reverse=True),
    )


def md_bold(text: str) -> str:
    """Convert **bold** markdown to HTML bold tags."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


# FIGURE

def render_figure(pil_img, heatmap, overlay_bgr, label, conf) -> io.BytesIO:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0a0e1a")
    accent = "#ef4444" if label == "Anomaly" else "#22c55e"
    for ax in axes:
        ax.set_facecolor("#0f1628")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d4a")
    axes[0].imshow(pil_img, cmap="gray")
    axes[0].set_title("Original X-Ray", color="#cbd5e1", fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")
    axes[1].imshow(cv2.resize(heatmap, (224, 224)), cmap="jet", interpolation="bilinear")
    axes[1].set_title("Grad-CAM Heatmap\n(Red = High Activation)", color="#cbd5e1", fontsize=12, fontweight="bold", pad=10)
    axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Prediction: {label}\nConfidence: {conf:.1%}", color=accent, fontsize=12, fontweight="bold", pad=10)
    axes[2].axis("off")
    plt.tight_layout(pad=2.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# PDF REPORT

def generate_pdf(label, conf, explanation, region_data, fig_buf, timestamp) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    W = A4[0] - 40*mm

    DARK   = colors.HexColor("#0a0e1a")
    BLUE   = colors.HexColor("#2563eb")
    CYAN   = colors.HexColor("#06b6d4")
    SLATE  = colors.HexColor("#64748b")
    LIGHT  = colors.HexColor("#0a1931")  
    CARD   = colors.HexColor("#0f1628")
    CARD2  = colors.HexColor("#111827")
    BORDER = colors.HexColor("#1e2d4a")
    RED    = colors.HexColor("#ef4444")
    GREEN  = colors.HexColor("#22c55e")
    ORANGE = colors.HexColor("#f97316")

    accent_color = RED if label == "Anomaly" else GREEN

    title_s = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=19,
        textColor=LIGHT, spaceAfter=4, alignment=TA_CENTER)
    
    sub_s   = ParagraphStyle("s", fontName="Helvetica", fontSize=9,
        textColor=colors.HexColor("#94a3b8"), spaceAfter=2, alignment=TA_CENTER)
    
    sec_s   = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=10,
        textColor=CYAN, spaceBefore=10, spaceAfter=4)
    
    body_s = ParagraphStyle("b", fontName="Helvetica", fontSize=11,
    textColor=colors.HexColor("#1a3a5c"), leading=16, spaceAfter=5)
   
    mono_s  = ParagraphStyle("m", fontName="Courier", fontSize=10,
        textColor=colors.HexColor("#94a3b8"), leading=14)
    
    disc_s  = ParagraphStyle("d", fontName="Helvetica-Oblique", fontSize=9.5,
        textColor=colors.HexColor("#fbbf24"), leading=14, alignment=TA_CENTER)

    story = []

   # Header
    story += [Spacer(1, 4*mm),
              Paragraph("XAI-ENHANCED X-RAY DIAGNOSTIC REPORT", title_s),
              Paragraph(timestamp, sub_s),
              Spacer(1, 3*mm),
              HRFlowable(width=W, color=BLUE, thickness=1.5),
              Spacer(1, 4*mm)]

    # Prediction table
    label_hex = "ef4444" if label == "Anomaly" else "22c55e"
    rows = [
        [Paragraph("CLASSIFICATION", mono_s), Paragraph(f'<font color="#{label_hex}"><b>{label}</b></font>', body_s)],
        [Paragraph("CONFIDENCE",     mono_s), Paragraph(f'<font color="#60a5fa"><b>{conf:.2%}</b></font>', body_s)],
        [Paragraph("MODEL",          mono_s), Paragraph("ResNet-18 + Grad-CAM (PyTorch)", body_s)],
        [Paragraph("PEAK REGION",    mono_s), Paragraph(region_data["peak"], body_s)],
        [Paragraph("HIGH ACT. AREA", mono_s), Paragraph(f'{region_data["high_ratio"]:.1%}', body_s)],
        [Paragraph("TIMESTAMP",      mono_s), Paragraph(timestamp, body_s)],
    ]
    t = Table(rows, colWidths=[44*mm, W - 44*mm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [CARD, CARD2]),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 8), ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 6), ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ]))
    story += [t, Spacer(1, 5*mm)]

    # Figure
    story += [Paragraph("VISUALISATION", sec_s),
              HRFlowable(width=W, color=BORDER, thickness=0.5),
              Spacer(1, 2*mm)]
    fig_buf.seek(0)
    story += [RLImage(fig_buf, width=W, height=W * (5/15)), Spacer(1, 5*mm)]

    # XAI Summary
    story += [Paragraph("XAI SUMMARY", sec_s),
              HRFlowable(width=W, color=BORDER, thickness=0.5),
              Spacer(1, 2*mm),
              Paragraph(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", explanation["simple"]), body_s),
              Spacer(1, 3*mm)]

    # Region table
    story += [Paragraph("REGION ACTIVATION BREAKDOWN", sec_s),
              HRFlowable(width=W, color=BORDER, thickness=0.5),
              Spacer(1, 2*mm)]

    reg_rows = [[Paragraph("<b>Region</b>", mono_s),
                 Paragraph("<b>Score</b>", mono_s),
                 Paragraph("<b>Level (bar)</b>", mono_s)]]
    for rname, rval in explanation["regions"]:
        bar   = "█" * int(rval * 22) + "░" * (22 - int(rval * 22))
        col   = "#ef4444" if rval > 0.3 else ("#f97316" if rval > 0.15 else "#475569")
        reg_rows.append([
            Paragraph(rname, body_s),
            Paragraph(f'<font name="Courier" color="#60a5fa">{rval:.4f}</font>', body_s),
            Paragraph(f'<font name="Courier" color="{col}" size="7">{bar}</font>', body_s),
        ])
    rt = Table(reg_rows, colWidths=[52*mm, 26*mm, W - 78*mm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [CARD, CARD2]),
        ("GRID", (0,0), (-1,-1), 0.4, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 7), ("RIGHTPADDING",  (0,0),(-1,-1), 7),
        ("TOPPADDING",   (0,0),(-1,-1), 5), ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    story += [rt, Spacer(1, 5*mm)]

    # Detailed sections
    for title, key in [
        ("DETAILED REGION ANALYSIS",       "region_detail"),
        ("HEATMAP PATTERN INTERPRETATION", "pattern"),
        ("CONFIDENCE INTERPRETATION",      "conf_interp"),
        ("CLINICAL CONTEXT",               "clinical"),
    ]:
        story += [Paragraph(title, sec_s),
                  HRFlowable(width=W, color=BORDER, thickness=0.5),
                  Spacer(1, 2*mm),
                  Paragraph(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", explanation[key]), body_s),
                  Spacer(1, 4*mm)]

    # Disclaimer
    story += [HRFlowable(width=W, color=BORDER, thickness=0.5), Spacer(1, 3*mm),
              Paragraph("DISCLAIMER: This report was generated by a research prototype. "
                        "It is NOT a substitute for professional medical diagnosis. "
                        "Always consult a qualified radiologist before making any clinical decision.", disc_s)]

    doc.build(story)
    buf.seek(0)
    return buf.read()


# SIDEBAR

with st.sidebar:
    st.markdown("""
    <div style='padding:6px 0 18px 0;'>
        <!-- CHANGED: font-size from 1.3rem to 1.6rem -->
        <p style='font-size:1.6rem;font-weight:700;margin:0;color:#f1f5f9;'>🩻 XAI X-Ray</p>
        <!-- CHANGED: font-size from 0.7rem to 0.85rem, color from #475569 to #94a3b8 -->
        <p style='font-size:0.85rem;color:#94a3b8;margin:3px 0 0 0;font-family:monospace;'>
            w1990606 · UoW Final Project
        </p>
    </div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["png","jpg","jpeg"])

    st.markdown("---")
    model_loaded = os.path.exists(MODEL_PATH)
    st.markdown(f"""
    <!-- CHANGED: label font-size from 0.65rem to 0.8rem -->
    <p style='font-size:0.8rem;color:#94a3b8;font-family:monospace;
              letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Model Info</p>
    <!-- CHANGED: font-size from 0.8rem to 0.95rem -->
    <div style='font-size:0.95rem;color:#cbd5e1;line-height:1.9;'>
        {"🟢 Weights loaded" if model_loaded else "🟡 ImageNet defaults"}<br>
        📐 ResNet-18 + Grad-CAM<br>
        ⚡ Device: <code style='color:#60a5fa;'>{DEVICE}</code><br>
        🏷 Classes: Anomaly / Normal
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <!-- CHANGED: font-size from 0.75rem to 0.9rem, color from #475569 to #94a3b8 -->
    <p style='font-size:0.9rem;color:#94a3b8;line-height:1.6;'>
        Addresses the AI "black box" problem in medical imaging by combining
        Grad-CAM heatmaps with region-aware XAI text explanations.
        Built for low-resource healthcare settings.
    </p>""", unsafe_allow_html=True)


# HEADER + MODEL INIT

model      = load_model()
cam_engine = GradCAM(model, model.layer4[-1].conv2)

st.markdown(f"""
<div class="dash-header">
  <div>
    <p class="dash-title">🩻 XAI-Enhanced X-Ray Diagnostics</p>
    <p class="dash-sub">Advancing Developing Countries' Medical Imaging Capabilities · ResNet-18 + Grad-CAM · PyTorch</p>
  </div>
  <div class="status-pill">
    <div class="status-dot"></div>System Ready · {DEVICE.upper()}
  </div>
</div>""", unsafe_allow_html=True)


# LANDING

if uploaded_file is None:
    c1, c2, c3 = st.columns(3)

    upload_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="52" height="52" viewBox="0 0 24 24" 
         fill="none" stroke="#60a5fa" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 3v13"/>
      <path d="M8 7l4-4 4 4"/>
      <path d="M4 14v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/>
    </svg>"""

    microscope_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="52" height="52" viewBox="0 0 100 100"
         fill="none" stroke="#60a5fa" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
      <rect x="38" y="8" width="14" height="22" rx="3"/>
      <circle cx="45" cy="34" r="6"/>
      <path d="M45 40 Q45 60 35 70"/>
      <rect x="20" y="62" width="30" height="5" rx="2"/>
      <rect x="18" y="85" width="44" height="7" rx="3"/>
      <line x1="45" y1="67" x2="45" y2="85"/>
      <rect x="40" y="40" width="10" height="10" rx="2"/>
      <line x1="35" y1="62" x2="30" y2="72"/>
      <line x1="30" y1="72" x2="30" y2="85"/>
    </svg>"""

    bulb_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="52" height="52" viewBox="0 0 24 24"
         fill="none" stroke="#60a5fa" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <!-- Bulb dome -->
      <path d="M9 21h6"/>
      <path d="M10 17h4"/>
      <!-- Main bulb shape -->
      <path d="M12 2a7 7 0 0 1 5 11.93V15a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1v-1.07A7 7 0 0 1 12 2z"/>
      <!-- Inner filament detail -->
      <path d="M10 13 Q12 11 14 13" stroke-width="1.4"/>
    </svg>"""

    for col, icon, title, desc in [
        (c1, upload_svg,     "Upload X-Ray",    "Use the sidebar to upload a PNG/JPG bone X-ray image."),
        (c2, microscope_svg, "AI Analysis",     "ResNet-18 classifies the image as Anomaly or Normal with a confidence score."),
        (c3, bulb_svg,       "XAI Explanation", "Grad-CAM heatmaps and region-aware text explain the model's reasoning — addressing the black box problem."),
    ]:
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Upload an X-ray image in the sidebar to begin analysis.")
    st.stop()


# ANALYSIS

pil_img   = Image.open(uploaded_file).convert("RGB")
x         = preprocess(pil_img)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with torch.no_grad():
    logits = model(x)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

pred_idx    = int(probs.argmax())
label       = CLASS_NAMES[pred_idx]
conf        = float(probs[pred_idx])
heatmap     = cam_engine.generate(x, pred_idx)
heatmap_224 = cv2.resize(heatmap, (224, 224))
overlay_bgr = make_overlay(pil_img, heatmap_224)
region_data = analyse_regions(heatmap_224)
explanation = build_explanation(label, conf, region_data)
fig_buf     = render_figure(pil_img, heatmap_224, overlay_bgr, label, conf)


# COLUMN LAYOUT

left, right = st.columns([3, 2], gap="large")

with left:

    # Prediction badge
    if label == "Anomaly":
        st.markdown("""
        <div class="badge-anomaly">
            <div class="badge-label" style="color:#ef4444;">⚠  ANOMALY DETECTED</div>
            <div class="badge-sub">Potential bone irregularity identified</div>
        </div>""", unsafe_allow_html=True)
        fill = "conf-fill-anom"
    else:
        st.markdown("""
        <div class="badge-normal">
            <div class="badge-label" style="color:#22c55e;">✓  NORMAL</div>
            <div class="badge-sub">No significant anomaly detected</div>
        </div>""", unsafe_allow_html=True)
        fill = "conf-fill-norm"

    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;font-size:0.72rem;
                color:#475569;margin:8px 0 2px 0;font-family:monospace;'>
        <span>Model Confidence</span>
        <span style='color:#60a5fa;font-weight:700;'>{conf:.2%}</span>
    </div>
    <div class="conf-track">
        <div class="{fill}" style="width:{conf*100:.1f}%;"></div>
    </div>""", unsafe_allow_html=True)

    # Figure
    st.image(fig_buf, use_container_width=True)

    # Metrics
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-val">{conf:.1%}</div>
            <div class="metric-lbl">Confidence</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{region_data['high_ratio']:.1%}</div>
            <div class="metric-lbl">High Activation Area</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{region_data['peak_val']:.2f}</div>
            <div class="metric-lbl">Peak Activation</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{DEVICE.upper()}</div>
            <div class="metric-lbl">Compute Device</div>
        </div>
    </div>""", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card-label">🧠 XAI Explanation</div>', unsafe_allow_html=True)

    # Simple summary — always visible
    box_cls = "xai-summary-warn" if label == "Anomaly" else "xai-summary"
    lbl_col = "#f97316"           if label == "Anomaly" else "#2563eb"
    st.markdown(f"""
    <div class="{box_cls}">
        <div class="xai-section-title" style="color:{lbl_col};">Summary</div>
        <div class="xai-text">{md_bold(explanation['simple'])}</div>
    </div>""", unsafe_allow_html=True)

    # Region activation bars
    st.markdown('<div class="card-label" style="margin-top:14px;">Region Activation</div>',
                unsafe_allow_html=True)
    for rname, rval in explanation["regions"]:
        pct   = int(rval * 100)
        bcol  = "#ef4444" if rval > 0.3 else ("#f97316" if rval > 0.15 else "#3b82f6")
        st.markdown(f"""
        <div class="region-row">
            <span class="region-name">{rname}</span>
            <div class="region-track">
                <div style="width:{pct}%;background:{bcol};height:100%;border-radius:99px;"></div>
            </div>
            <span class="region-val">{rval:.3f}</span>
        </div>""", unsafe_allow_html=True)

    # Detailed expandable sections
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📍 Detailed Region Analysis"):
        st.markdown(f"""
        <div class="xai-section">
            <div class="xai-section-title" style="color:#06b6d4;">Region Analysis</div>
            <div class="xai-text">{md_bold(explanation['region_detail'])}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("🌡 Heatmap Pattern Interpretation"):
        st.markdown(f"""
        <div class="xai-section">
            <div class="xai-section-title" style="color:#06b6d4;">Activation Pattern</div>
            <div class="xai-text">{md_bold(explanation['pattern'])}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("📊 Confidence Interpretation"):
        st.markdown(f"""
        <div class="xai-section">
            <div class="xai-section-title" style="color:#06b6d4;">Confidence Score</div>
            <div class="xai-text">{md_bold(explanation['conf_interp'])}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("🏥 Clinical Context"):
        st.markdown(f"""
        <div class="xai-section">
            <div class="xai-section-title" style="color:#06b6d4;">Medical Context</div>
            <div class="xai-text">{md_bold(explanation['clinical'])}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠ <b>Disclaimer:</b> Research prototype only. Not a substitute for professional medical
        diagnosis. Consult a qualified radiologist before any clinical decision.
    </div>""", unsafe_allow_html=True)

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-label">📥 Export</div>', unsafe_allow_html=True)

    pdf_bytes = generate_pdf(label, conf, explanation, region_data, fig_buf, timestamp)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"xai_report_{ts}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl2:
        img_buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(img_buf, format="PNG")
        img_buf.seek(0)
        st.download_button(
            "🖼 Download Heatmap",
            data=img_buf,
            file_name=f"gradcam_{ts}.png",
            mime="image/png",
            use_container_width=True,
        )


# FOOTER

st.markdown("""
<div style='text-align:center;padding:32px 0 12px 0;color:#1e2d4a;
            font-size:0.7rem;font-family:monospace;'>
    XAI-Enhanced X-Ray Diagnostics · Shayne Felicien-Brown (w1990606) ·
    University of Westminster · BSc Computer Science Final Project
</div>""", unsafe_allow_html=True)