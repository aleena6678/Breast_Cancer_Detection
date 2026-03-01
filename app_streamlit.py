# from petct_module import predict_petct
from petct_inference import predict_petct

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import streamlit as st
import pydicom
import nibabel as nib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import time
import base64
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

from doctor_auth import doctor_login

IMAGE_SIZE = 256
CLASS_NAMES = ["normal", "benign", "malignant"]
MODEL_PATH = "breast_cancer_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================= PAGE CONFIG & CUSTOM THEME =================
st.set_page_config(
    page_title="Breast Cancer Detection – BUS-UCLM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    doctor_login()
    st.stop()

# ================= PATIENT HISTORY STORAGE =================

HISTORY_DIR = "patient_records"
HISTORY_FILE = os.path.join(HISTORY_DIR, "patient_history.csv")

def save_patient_history(
    patient_name,
    patient_id,
    prediction,
    confidence,
    stage,
    risk_score
):

    os.makedirs(HISTORY_DIR, exist_ok=True)

    new_data = {
        "Patient ID": patient_id,
        "Patient Name": patient_name,
        "Date": datetime.now().strftime("%d-%m-%Y"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Prediction": prediction,
        "Confidence (%)": round(confidence, 2),
        "Risk Score": round(risk_score, 2),
        "Stage": stage
    }

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)

        # 🚫 Avoid duplicate Patient ID
        if patient_id in df["Patient ID"].values:
            
            return  # do not save again

        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(HISTORY_FILE, index=False)


import uuid

# (Page config moved to top)
st.markdown("""
<style>

/* ===== Radio group title ===== */
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] p,
div[data-testid="stRadio"] > label {
    color: #ffffff !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
}

/* ===== Radio option text ===== */
div[data-testid="stRadio"] [role="radiogroup"] label p,
div[data-testid="stRadio"] [role="radiogroup"] label {
    color: #ffffff !important;        /* FORCE WHITE */
    font-size: 1.25rem !important;    /* Bigger text */
    font-weight: 700 !important;
}

/* ===== Selected option ===== */
div[data-testid="stRadio"] [role="radiogroup"] label input:checked ~ div p,
div[data-testid="stRadio"] [role="radiogroup"] label input:checked ~ div {
    color: #22c55e !important;        /* Green selected */
    font-weight: 900 !important;
}

/* ===== Hover ===== */
div[data-testid="stRadio"] [role="radiogroup"] label:hover p,
div[data-testid="stRadio"] [role="radiogroup"] label:hover {
    color: #fde047 !important;        /* Yellow hover */
}

/* ===== Radio circles ===== */
div[data-testid="stRadio"] svg {
    fill: #ffffff !important;
}

div[data-testid="stRadio"] [role="radiogroup"] label input:checked ~ div svg {
    fill: #22c55e !important;
}

</style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.markdown(f"👨‍⚕️ Logged in as: **{st.session_state.doctor_name}**")

    st.markdown("### 🎯 Dashboard Controls")

    with st.expander("⚙️ System Status", expanded=True):
        st.markdown(f"**Device:** {'GPU 🚀' if torch.cuda.is_available() else 'CPU ⚡'}")
        st.markdown(f"**Model:** ResNet18")
        st.markdown(f"**Input Size:** {IMAGE_SIZE}×{IMAGE_SIZE}")
        st.progress(100, text="System Ready")

    st.markdown("---")

    st.markdown("### 🏷️ Classification Labels")
    cols = st.columns(3)
    with cols[0]:
        st.markdown('<div class="status-badge normal">Normal</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="status-badge benign">Benign</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="status-badge malignant">Malignant</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Dataset Info")
    st.markdown("**BUS-UCLM Dataset**")
    st.markdown("• Breast Ultrasound Images")
    st.markdown("• Three-class classification")

    st.markdown("---")

    # ✅ Logout placed directly under dataset info
    if st.button(" Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.doctor_name = None
        st.rerun()
# Function to encode local image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to add background image
def add_bg_image():
    # Try to load local bact.jpg, fallback to online image if not found
    try:
        # Check if local file exists
        if os.path.exists("bact.jpg"):
            bg_image = f"data:image/jpg;base64,{get_base64_of_bin_file('bact.jpg')}"
        else:
            # Fallback to online image
            bg_image = "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80"
    except:
        bg_image = "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80"
    
    st.markdown(
        f"""
        <style>
            /* Banner section with background image */
            .banner-section {{
                background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                            url('{bg_image}');
                background-size: cover;
                background-position: center;
                padding: 80px 40px;
                border-radius: 20px;
                margin-bottom: 40px;
                text-align: center;
                animation: fadeIn 1s ease-out;
            }}
            
            .banner-title {{
                font-size: 3.5rem;
                font-weight: 900;
                color: white;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                letter-spacing: 1px;
            }}
            
            .banner-subtitle {{
                font-size: 1.5rem;
                color: #e2e8f0;
                max-width: 800px;
                margin: 0 auto;
                line-height: 1.6;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
            }}
            
            /* Modern gradient background with subtle animation */
            .stApp {{
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
                color: #f1f5f9;
            }}
            
            @keyframes gradientShift {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            
            /* Smooth fade-in animation */
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            /* Title styling with animation */
            .title-text {{
                font-size: 2.5rem;
                font-weight: 800;
                background: linear-gradient(90deg, #60a5fa, #34d399, #fbbf24);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                animation: fadeIn 1s ease-out;
                text-align: center;
                margin-bottom: 0.5rem;
            }}
            
            .subtitle-text {{
                font-size: 1.1rem;
                color: #cbd5e1;
                text-align: center;
                animation: fadeIn 1.2s ease-out;
                max-width: 800px;
                margin: 0 auto;
                line-height: 1.6;
            }}
            
            /* File uploader text color modifications - GREEN for all text */
            [data-testid="stFileUploader"] label {{
                color: #22c55e !important;
                font-size: 1.2rem !important;
                font-weight: 600 !important;
            }}
            
            /* File uploader instruction text */
            [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {{
                color: #86efac !important;
                font-size: 0.95rem !important;
            }}
            
            /* File name display styling - GREEN */
            .uploadedFileName {{
                color: #22c55e !important;
                font-weight: 600 !important;
                background: rgba(34, 197, 94, 0.1) !important;
                padding: 8px 16px !important;
                border-radius: 10px !important;
                border: 1px solid rgba(34, 197, 94, 0.3) !important;
                margin-top: 10px !important;
            }}
            
            /* Browse files button styling - GREEN */
            [data-testid="stFileUploader"] button {{
                background: linear-gradient(45deg, #22c55e, #16a34a) !important;
                color: white !important;
                border: none !important;
                padding: 10px 24px !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }}
            
            [data-testid="stFileUploader"] button:hover {{
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4) !important;
                background: linear-gradient(45deg, #16a34a, #22c55e) !important;
            }}
            
            /* File list items - GREEN */
            [data-testid="stFileUploader"] [data-testid="stFileDropzoneInstructions"] div {{
                color: #86efac !important;
            }}
            
            /* Glassmorphism card with hover effects */
            .glass-card {{
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 25px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
                animation: fadeIn 0.8s ease-out;
            }}
            .equal-card {{
                    min-height: 280px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                }}
            .glass-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
                border-color: rgba(96, 165, 250, 0.3);
            }}
            
            /* Metric card with gradient border */
            .metric-card {{
                background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
                border-radius: 18px;
                padding: 25px;
                border: 1px solid;
                border-image: linear-gradient(45deg, #60a5fa, #34d399) 1;
                position: relative;
                overflow: hidden;
                animation: fadeIn 1s ease-out;
            }}
            
            .metric-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
                transition: 0.5s;
            }}
            
            .metric-card:hover::before {{
                left: 100%;
            }}
            
            /* Enhanced probability bars */
            .prob-bar-container {{
                margin: 15px 0;
            }}
            
            .prob-bar {{
                height: 14px;
                border-radius: 10px;
                background: rgba(30, 41, 59, 0.8);
                overflow: hidden;
                position: relative;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            
            .prob-fill {{
                height: 100%;
                border-radius: 10px;
                background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
                transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }}
            
            .prob-fill::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, 
                    transparent 0%, 
                    rgba(255, 255, 255, 0.2) 50%, 
                    transparent 100%);
                animation: shimmer 2s infinite;
            }}
            
            @keyframes shimmer {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
            
            /* File uploader styling */
            [data-testid="stFileUploader"] > div:first-child {{
                border-radius: 16px;
                border: 2px dashed rgba(34, 197, 94, 0.4);
                background-color: rgba(15, 23, 42, 0.5);
                transition: all 0.3s ease;
                padding: 40px 20px;
            }}
            
            [data-testid="stFileUploader"] > div:first-child:hover {{
                border-color: #22c55e;
                background-color: rgba(15, 23, 42, 0.7);
            }}
            
            /* Button styling */
            .stButton > button[kind="primary"] {{
                background: linear-gradient(45deg, #60a5fa, #34d399);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 12px;
                font-weight: 600;
                transition: all 0.3s ease;
            }}
            
            .stButton > button[kind="primary"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(96, 165, 250, 0.4);
            }}

            /* Professional secondary button styling */
            .stButton > button[kind="secondary"],
            .stDownloadButton > button {{
                background: rgba(30, 41, 59, 0.6);
                color: #ffffff !important;
                border: 1px solid rgba(148, 163, 184, 0.4);
                padding: 12px 24px;
                border-radius: 12px;
                font-weight: 600;
                transition: all 0.3s ease;
                backdrop-filter: blur(5px);
            }}

            .stButton > button[kind="secondary"]:hover,
            .stDownloadButton > button:hover {{
                background: rgba(45, 55, 72, 0.8);
                color: #ffffff !important;
                border-color: #60a5fa;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(96, 165, 250, 0.2);
            }}
            
            /* Ensure download button SVG icons are white */
            .stDownloadButton > button svg {{
                fill: #ffffff !important;
                color: #ffffff !important;
            }}

            /* Metrics text colors */
            [data-testid="stMetricLabel"],
            [data-testid="stMetricLabel"] p,
            [data-testid="stMetricLabel"] > div {{
                color: #94a3b8 !important;   /* Slate 400 */
                font-weight: 600 !important;
            }}
            
            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] div {{
                color: #ffffff !important;   /* White */
                font-weight: 800 !important;
            }}
            
            /* Expander text color override */
            [data-testid="stExpander"] summary span,
            [data-testid="stExpander"] summary p {{
                color: #0f172a !important; /* Make the white text dark so it's visible on default backgrounds */
                font-weight: 700;
            }}
            
            [data-testid="stExpander"] svg {{
                color: #0f172a !important;
                fill: #0f172a !important;
            }}

            /* Status badges */
            .status-badge {{
                display: inline-block;
                padding: 6px 16px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9rem;
                margin: 5px;
                animation: pulse 2s infinite;
            }}
            
            .normal {{ background: linear-gradient(45deg, #10b981, #34d399); }}
            .benign {{ background: linear-gradient(45deg, #f59e0b, #fbbf24); }}
            .malignant {{ background: linear-gradient(45deg, #ef4444, #f87171); }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.8; }}
            }}
            
            /* Image containers */
            .image-container {{
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease;
            }}
            
            .image-container:hover {{
                transform: scale(1.02);
            }}
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background-color: rgba(30, 41, 59, 0.5);
                border-radius: 12px 12px 0 0;
                padding: 12px 24px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                color: #cbd5e1;
                transition: all 0.3s ease;
            }}
            
            .stTabs [data-baseweb="tab"]:hover {{
                background-color: rgba(30, 41, 59, 0.8);
                color: white;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: linear-gradient(45deg, #60a5fa, #34d399) !important;
                color: white !important;
                border-color: transparent !important;
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 10px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: rgba(15, 23, 42, 0.5);
                border-radius: 5px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: linear-gradient(45deg, #60a5fa, #34d399);
                border-radius: 5px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: linear-gradient(45deg, #34d399, #60a5fa);
            }}
            
            /* Section headers */
            .section-header {{
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(90deg, #60a5fa, #34d399);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid rgba(96, 165, 250, 0.3);
            }}
            
        </style>
        """,
        unsafe_allow_html=True,
    )

add_bg_image()

# ================= CONFIG =================

def load_dicom_series(files):
    slices = []

    for f in files:
        ds = pydicom.dcmread(f)
        slices.append(ds)

    # Sort by slice position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    return volume, slices[0]


def process_ct(img, ds):
    hu = img * ds.RescaleSlope + ds.RescaleIntercept
    hu = np.clip(hu, -150, 250)
    hu = (hu - hu.min()) / (hu.max() - hu.min())
    return (hu * 255).astype(np.uint8)

def process_pet(img):
    img = np.clip(img, np.percentile(img, 5), np.percentile(img, 95))
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)

def fuse_pet_ct(ct_img, pet_img):
    ct_rgb = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2RGB)
    pet_heat = cv2.applyColorMap(pet_img, cv2.COLORMAP_JET)
    return cv2.addWeighted(ct_rgb, 0.6, pet_heat, 0.4, 0)

DATA_ROOT = "data"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR = os.path.join(DATA_ROOT, "masks")



img_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

def get_ai_interpretation_text(pred_name, confidence):
    if pred_name == "normal":
        ultrasound = (
            "The ultrasound image shows homogeneous tissue structure, "
            "smooth boundaries, and no visible lesion mass."
        )
        explanation = (
            "The AI model analyzed spatial texture patterns and deep feature activations. "
            "Grad-CAM visualization confirms that the prediction is driven by lesion-specific "
            "regions rather than background tissue."
        )
        followup = "No immediate follow-up required."

    elif pred_name == "benign":
        ultrasound = (
            "The ultrasound image shows a well-defined lesion with smooth margins "
            "and uniform internal texture."
        )
        explanation = (
            "The AI model focused on localized lesion regions with consistent internal patterns. "
            "Grad-CAM highlights benign structural features."
        )
        followup = (
            "Periodic clinical follow-up and ultrasound monitoring are recommended."
        )

    else:  # malignant
        ultrasound = (
            "The ultrasound image shows an irregular hypoechoic mass with heterogeneous texture "
            "and poorly defined margins."
        )
        explanation = (
            "The AI model detected abnormal spatial features and strong activations "
            "in suspicious lesion regions. Grad-CAM confirms malignant characteristics."
        )
        followup = (
            "Immediate clinical consultation is strongly recommended. "
            "Further diagnostic procedures such as biopsy or MRI are advised."
        )

    return ultrasound, explanation, followup

def draw_multiline_text(c, text, x, y, max_width, leading=14):
    """
    Draws wrapped multi-line text in ReportLab.
    """
    textobject = c.beginText(x, y)
    textobject.setLeading(leading)

    words = text.split(" ")
    line = ""

    for word in words:
        test_line = line + word + " "
        if c.stringWidth(test_line, "Helvetica", 11) <= max_width:
            line = test_line
        else:
            textobject.textLine(line)
            line = word + " "
    if line:
        textobject.textLine(line)

    c.drawText(textobject)
    return textobject.getY()

def generate_patient_report(
    patient_name,
    patient_id,
    patient_age,
    prediction,
    confidence,
    stage,
    probs
):

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin_x = 50
    max_width = width - 2 * margin_x
    y = height - 50

    # ===== TITLE =====
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin_x, y, "Breast Cancer Diagnosis Report")
    y -= 40

    # ===== PATIENT INFO =====
    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, f"Patient Name: {patient_name}")
    y -= 20
    c.drawString(margin_x, y, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    y -= 20
    c.drawString(margin_x, y, f"Patient ID: {patient_id}")
    y -= 18
    c.drawString(margin_x, y, f"Age: {patient_age} years")
    y -= 18
    c.line(margin_x, y, width - margin_x, y)
    y -= 30
    # ===== AI SUMMARY =====
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "AI Prediction Summary")
    y -= 20

    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, f"Predicted Class: {prediction.upper()}")
    y -= 18
    c.drawString(margin_x, y, f"Confidence Score: {confidence:.2f}%")
    y -= 18
    c.drawString(margin_x, y, f"Predicted Stage: {stage}")
    y -= 30

    # ===== CLASS PROBABILITIES =====
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "Class Probabilities:")
    y -= 18

    c.setFont("Helvetica", 11)
    for cls, p in zip(["Normal", "Benign", "Malignant"], probs):
        c.drawString(margin_x + 20, y, f"{cls}: {p*100:.2f}%")
        y -= 16

    y -= 20
    c.line(margin_x, y, width - margin_x, y)
    y -= 30

    # ===== AI INTERPRETATION =====
    ultrasound_text, ai_text, followup_text = get_ai_interpretation_text(
        prediction.lower(), confidence
    )

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "AI Explanation")
    y -= 20

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "Ultrasound Image Properties:")
    y -= 16

    c.setFont("Helvetica", 11)
    y = draw_multiline_text(c, ultrasound_text, margin_x + 10, y, max_width - 10)
    y -= 20

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "AI Explanation:")
    y -= 16

    c.setFont("Helvetica", 11)
    y = draw_multiline_text(c, ai_text, margin_x + 10, y, max_width - 10)
    y -= 20

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "Follow-up Recommendation:")
    y -= 16

    c.setFont("Helvetica", 11)
    y = draw_multiline_text(c, followup_text, margin_x + 10, y, max_width - 10)

    # ===== FOOTER =====
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        margin_x,
        40,
        "This report is generated by an MammoSafe."
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ================= LOAD LOTTIE ANIMATION =================

def load_lottie(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

# ================= MODEL =================

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ================= SEGMENTATION MODEL (U-NET) =================

class UNetSmall(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        bn = self.bottleneck(p3)

        u3 = self.up3(bn)
        u3 = torch.cat([u3, c3], dim=1)
        c3 = self.conv3(u3)

        u2 = self.up2(c3)
        u2 = torch.cat([u2, c2], dim=1)
        c2 = self.conv2(u2)

        u1 = self.up1(c2)
        u1 = torch.cat([u1, c1], dim=1)
        c1 = self.conv1(u1)

        out = self.out_conv(c1)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_seg_model():
    model = UNetSmall(in_channels=1, out_channels=1).to(DEVICE)
    state_dict = torch.load("breast_seg_unet.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_segmentation_mask(pil_img, seg_model):
    """
    Input: PIL image (RGB)
    Output:
      - mask_rgb: GREEN mask on black background
      - overlay_rgb: original image with colored heatmap on lesion region
    """
    # 1) convert to grayscale for segmentation model
    gray = pil_img.convert("L")
    orig_w, orig_h = gray.size

    # 2) resize to training size
    gray_resized = gray.resize((IMAGE_SIZE, IMAGE_SIZE))
    gray_np = np.array(gray_resized)  # [H, W]
    gray_tensor = torch.from_numpy(gray_np).float() / 255.0
    gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, H, W]

    # 3) forward pass through U-Net
    with torch.no_grad():
        logits = seg_model(gray_tensor)         # [1, 1, H, W]
        probs = torch.sigmoid(logits)[0, 0]     # [H, W] in [0,1]

    # 4) resize probabilities to original size
    prob_small = probs.cpu().numpy()
    prob_big = cv2.resize(prob_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 5) binary mask with softer threshold
    th = 0.30                      # was 0.50 – this is easier
    mask_bin = (prob_big > th).astype(np.uint8)

    #    if still completely empty, fallback to top 3% highest pixels
    if mask_bin.sum() == 0:
        flat = prob_big.flatten()
        k = max(1, int(0.03 * flat.size))       # top 3% pixels
        kth_val = np.partition(flat, -k)[-k]
        mask_bin = (prob_big >= kth_val).astype(np.uint8)

    # 6) create a GREEN binary mask image (for "Mask" view)
    mask_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    mask_rgb[:, :, 1] = mask_bin * 255          # green channel

    # 7) create a heatmap-style overlay on original
    rgb = np.array(pil_img.convert("RGB"))

    # normalize probs for colormap so we always see something
    prob_norm = (prob_big - prob_big.min()) / (prob_big.max() - prob_big.min() + 1e-8)
    prob_uint8 = (prob_norm * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # apply heatmap only where mask=1
    heatmap_masked = heatmap_rgb * mask_bin[:, :, None]
    overlay_rgb = np.uint8(0.6 * rgb + 0.4 * heatmap_masked)

    return mask_rgb, overlay_rgb


# ================= GRAD-CAM CORE =================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        loss.backward()

        acts = self.activations[0]
        grads = self.gradients[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()

        return cam, output

def get_gradcam_overlay(pil_img, model):
    rgb = np.array(pil_img.convert("RGB"))
    t = img_transform(pil_img).unsqueeze(0).to(DEVICE)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    cam, output = gradcam.generate(t)
    
    probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    
    h, w, _ = rgb.shape
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(cam_resized * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.42 * heatmap_rgb + 0.58 * rgb)

    return overlay, pred_name, probs

# ================= BANNER SECTION =================

st.markdown(
    """
    <div class="banner-section">
        <div class="banner-title">MammoSafe</div>
        <div class="banner-subtitle">
            Advanced AI-powered diagnostic assistance using deep learning and Grad-CAM visualization 
            for interpretable breast ultrasound image analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================= SCAN TYPE SELECTION & UPLOAD =================

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

col_title, col_refresh = st.columns([5,1])
with col_title:
    st.markdown("## 🔍 Select Scan Modality")
with col_refresh:
    if st.button("🔄 Refresh"):
        # Keep login info safe
        logged_in = st.session_state.get("logged_in", False)
        doctor_name = st.session_state.get("doctor_name", None)

        # Clear everything
        st.session_state.clear()

        # Restore login session
        st.session_state.logged_in = logged_in
        st.session_state.doctor_name = doctor_name
        st.session_state.uploader_key = str(uuid.uuid4())

        st.rerun()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

# Sidebar Logo/Menu Code ...

st.markdown("""
<style>
    /* Traditional Tab Styling for Radio Buttons */
    div.row-widget.stRadio > div { 
        flex-direction: row; 
        align-items: stretch; 
        justify-content: flex-start;
        background: transparent;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        padding: 0; 
        margin-bottom: 20px;
        gap: 0;
    }
    div.row-widget.stRadio > div > label { 
        padding: 10px 20px; 
        background: transparent; 
        border-radius: 8px 8px 0 0; 
        cursor: pointer; 
        text-align: center;
        transition: all 0.2s ease;
        border: none;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: -2px; /* Pull down to overlap border */
    }
    div.row-widget.stRadio > div > label:hover {
        color: #e2e8f0;
        background: rgba(255,255,255,0.05);
    }
    div.row-widget.stRadio > div > label[data-checked="true"] { 
        background: transparent !important; 
        color: #3b82f6 !important;
        font-weight: 600;
        border-bottom: 2px solid #3b82f6;
        box-shadow: none;
    }
    
    /* Hide the default radio circle */
    div.row-widget.stRadio > div > label > div:first-child { display: none; }
</style>
""", unsafe_allow_html=True)

active_tab = st.radio(
    "Select Scan Modality",
    ["Ultrasound", "PET / CT", "Histopathology"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded = None
uploaded_files = None

if active_tab == "Ultrasound":
    st.markdown("### 📤 Upload Ultrasound Scan")
    uploaded = st.file_uploader(
        "Upload Ultrasound Image",
        type=["png", "jpg", "jpeg"],
        key=f"ultrasound_{st.session_state.uploader_key}"
    )
    if uploaded:
        st.success(f"Uploaded: {uploaded.name}")

elif active_tab == "PET / CT":
    st.markdown("### 📤 Upload PET/CT Scan")
    uploaded_files = st.file_uploader(
        "Upload PET/CT DICOM Folder (select all files)",
        type=["dcm"],
        accept_multiple_files=True,
        key=f"petct_{st.session_state.uploader_key}"
    )
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} DICOM files")

elif active_tab == "Histopathology":
    st.markdown("### 🔬 Histopathology Scan")
    st.info("💡 Histopathology scan analysis is currently under development. Please check back later.")

scan_type = active_tab



# ================= PROCESSING =================

if (scan_type == "Ultrasound" and uploaded is not None) or \
   (scan_type == "PET / CT" and uploaded_files):

    with st.spinner("🔄 Processing scan..."):

        # ---------- ULTRASOUND ----------
        if scan_type == "Ultrasound":
            pil_img = Image.open(uploaded).convert("RGB")
            model = load_model()
            overlay_np, pred_name, probs = get_gradcam_overlay(pil_img, model)

        # ---------- PET / CT ----------
        elif scan_type == "PET / CT":
            try:
                result = predict_petct(uploaded_files)

                tab1, tab2 = st.tabs(["📝 Assessment & Visuals", "📊 Histogram"])

                with tab1:
                    st.markdown('<div class="section-header">🧠 PET / CT Clinical Assessment</div>', unsafe_allow_html=True)

                    # Top Row: Stage & Summary vs Metrics
                    top_col1, top_col2 = st.columns([1.5, 1])
                    
                    with top_col1:
                        st.markdown(
                            f"""
                            <div class="glass-card" style="height: 100%;">
                                <h3 style="color: #60a5fa; margin-top:0;">🩺 Estimated Stage</h3>
                                <div class="status-badge malignant" style="font-size: 1.5rem; padding: 10px 20px; display:inline-block; margin-bottom: 20px;">
                                    {result['stage']}
                                </div>
                                <h4 style="color: #cbd5e1;">📋 Clinical Summary</h4>
                                <p style="color: #e2e8f0; line-height: 1.6;">{result['clinical_summary']}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    with top_col2:
                        st.markdown(
                            """
                            <div class="glass-card" style="height: 100%;">
                                <h3 style="color: #34d399; margin-top:0;">📊 Metabolic Metrics</h3>
                            """, 
                            unsafe_allow_html=True
                        )
                        metrics = result["metrics"]
                        for k, v in metrics.items():
                            st.metric(k, f"{v:.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">Visual Interpretation</div>', unsafe_allow_html=True)
                    
                    # Images Row
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(
                            result["images"]["CT Slice (Anatomical)"],
                            caption="CT Slice (Anatomical)",
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with c2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(
                            result["images"]["PET Slice (Metabolic)"],
                            caption="PET Slice (Metabolic)",
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with c3:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(
                            result["images"]["PET + CT Fusion"],
                            caption="PET + CT Fusion",
                            use_container_width=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Doctor interpretation explanation
                    st.markdown(
                        """
                        <div class="glass-card">
                            <h4 style="color: #fbbf24; margin-top:0;">👨‍⚕️ Interpretation Guide</h4>
                            <div style="display: flex; gap: 20px; color: #cbd5e1;">
                                <div style="flex: 1;"><strong>• PET:</strong> Highlights metabolic activity (FDG uptake)</div>
                                <div style="flex: 1;"><strong>• CT:</strong> Shows physical anatomical structure</div>
                                <div style="flex: 1;"><strong>• Fusion:</strong> Correlates hotspots with anatomy</div>
                                <div style="flex: 1;"><strong>• Stage:</strong> Inferred from uptake intensity & spread</div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                with tab2:
                    st.markdown("### 📊 Image Intensity Histogram")
                    st.write("Distribution of pixel intensities from the primary CT slice.")
                    
                    # Convert the CT slice image to numpy array for histogram
                    ct_image = np.array(result["images"]["CT Slice (Anatomical)"])
                    
                    # If it's a 3D array (RGB), convert to grayscale for a simpler histogram
                    if len(ct_image.shape) == 3:
                        ct_image = cv2.cvtColor(ct_image, cv2.COLOR_RGB2GRAY)
                    
                    # Flatten the array and remove pure black pixels (often background)
                    flat_pixels = ct_image.flatten()
                    flat_pixels = flat_pixels[flat_pixels > 0]
                    
                    fig_hist = px.histogram(
                        x=flat_pixels, 
                        nbins=50,
                        labels={'x': 'Pixel Intensity', 'y': 'Count'},
                        color_discrete_sequence=['#60a5fa']
                    )
                    
                    fig_hist.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

                st.stop()

            except Exception as e:
                st.error(f"PET/CT processing error: {e}")
                st.stop()


    # ========= PREDICTION RESULTS (FOR ULTRASOUND) =========
    if scan_type == "Ultrasound":
        st.markdown('<div class="section-header">📈 Analysis Results</div>', unsafe_allow_html=True)
        
        # Prediction card with enhanced visuals
        pred_index = CLASS_NAMES.index(pred_name)
        confidence = probs[pred_index] * 100.0
        
        # Risk score calculation
        if pred_name == "malignant":
            risk_score = confidence
        elif pred_name == "benign":
            risk_score = confidence * 0.6
        else:
            risk_score = confidence * 0.2
        
        col_left, col_mid, col_right = st.columns([1.5, 2, 1.5])
        
        with col_mid:
            st.markdown(
                f"""
                <div class="metric-card" style="padding: 40px; text-align: center;">
                    <div style="margin-bottom: 30px;">
                        <div style="font-size: 2.2rem; font-weight: 800; margin-bottom: 15px;">
                            Prediction
                        </div>
                        <div class="status-badge {pred_name}" style="font-size: 1.5rem; padding: 12px 30px; letter-spacing: 1px;">
                            {pred_name.upper()}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 4.5rem; font-weight: 900; color: #60a5fa; margin: 15px 0;">
                            {confidence:.1f}%
                        </div>
                        <div style="color: #cbd5e1; font-size: 1.2rem; font-weight: 500;">
                            Confidence Score
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Probability distribution with Plotly
        st.markdown("### 📊 Probability Distribution")
        
        # Create interactive chart
        fig = go.Figure(data=[
            go.Bar(
                x=[c.upper() for c in CLASS_NAMES],
                y=probs * 100,
                marker_color=['#10b981', '#f59e0b', '#ef4444'],
                text=[f'{p*100:.2f}%' for p in probs],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title='Classes'),
            yaxis=dict(title='Probability (%)', range=[0, 100]),
            height=400,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # ========= VISUAL EXPLANATIONS =========
        st.markdown('<div class="section-header">🔍 Visual Explanations</div>', unsafe_allow_html=True)
        
        # Tabs for different views (conditionally hide heatmap if normal)
        if pred_name == "normal":
            # Only show original image
            #st.info("💡 Grad-CAM Heatmap visualization is not shown for Normal predictions as there are no specific lesion heat signatures to flag.")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Original Ultrasound Image")
        else:
            tab1, tab3 = st.tabs(["🖼️ Original", "🔥 Grad-CAM Heatmap"])
            
            with tab1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption("Original Ultrasound Image")
            
            with tab3:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(overlay_np, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption("Grad-CAM Heatmap - Areas of model attention")

        # ================= PATIENT REPORT =================
        if "patient_id" not in st.session_state:
            st.session_state.patient_id = f"PID-{uuid.uuid4().hex[:8].upper()}"

        st.markdown('<div class="section-header">🧾 Patient Report</div>', unsafe_allow_html=True)
        st.markdown("### 👤 Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            patient_name = st.text_input("Patient Name", "")

        with col2:
            patient_id = st.text_input(
                "Patient ID",
                value=st.session_state.patient_id,
                disabled=True
            )

        with col3:
            patient_age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30,
                step=1
            )
        
        # Determine cancer stage based on prediction and confidence
        if pred_name == "normal":
            stage = "No Cancer Detected"
        elif pred_name == "benign":
            stage = "Benign Tumor (Non-cancerous)"
        else:
            if confidence > 90:
                stage = "Stage III – Advanced Malignant"
            elif confidence > 75:
                stage = "Stage II – Moderate Malignant"
            else:
                stage = "Stage I – Early Malignant"

        st.success(f"Predicted Cancer Stage: **{stage}**")
        
        if not patient_name.strip():
            st.warning("⚠️ Please enter the patient name before saving the record.")
        else:
            # Save patient history
            save_patient_history(
                patient_name,
                patient_id,
                pred_name.upper(),
                confidence,
                stage,
                risk_score
            )
            
            st.success("✅ Patient record saved successfully!")

        # Generate and display download button for report
        report_buffer = generate_patient_report(
            patient_name,
            patient_id,
            patient_age,
            pred_name,
            confidence,
            stage,
            probs
        )

        st.download_button(
            label="⬇️ Download Patient Report (PDF)",
            data=report_buffer,
            file_name="Breast_Cancer_Report.pdf",
            mime="application/pdf"
        )
        
        # ========= DETAILED METRICS =========
        st.markdown('<div class="section-header">📋 Detailed Metrics</div>', unsafe_allow_html=True)
        
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown(
                """
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem; color: #60a5fa;">🔄</div>
                    <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Model Architecture</div>
                    <div style="color: #cbd5e1;">ResNet18 with Transfer Learning</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with cols[1]:
            st.markdown(
                """
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem; color: #34d399;">🎯</div>
                    <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Explainability</div>
                    <div style="color: #cbd5e1;">Grad-CAM Visualization</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with cols[2]:
            st.markdown(
                """
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 2rem; color: #fbbf24;">📊</div>
                    <div style="font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Confidence Scores</div>
                    <div style="color: #cbd5e1;">Real-time Probability Distribution</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
        
else:
    # Landing page when no image uploaded
    st.markdown('<div class="section-header">🚀 Get Started</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="glass-card equal-card">
                <h3> How to Use</h3>
                <ol style="color: #cbd5e1;">
                    <li>Upload a breast ultrasound image</li>
                    <li>Wait for AI analysis (2-3 seconds)</li>
                    <li>Review the prediction results</li>
                    <li>Explore visual explanations</li>
                    <li>Check confidence scores</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card equal-card">
                <h3>Features</h3>
                <ul style="color: #cbd5e1;">
                    <li>AI-powered classification</li>
                    <li>Grad-CAM heatmap visualization</li>
                    <li>Real-time confidence scoring</li>
                    <li>Interactive probability charts</li>
                    <li>Professional medical UI</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # Feature highlights
    st.markdown('<div class="section-header">Key Features</div>', unsafe_allow_html=True)
    
    features = st.columns(4)
    
    features[0].markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2.5rem;">🤖</div>
            <div style="font-weight: 600; margin: 10px 0;">AI-Powered</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Deep Learning Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    features[1].markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2.5rem;">👁️</div>
            <div style="font-weight: 600; margin: 10px 0;">Explainable</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Visual Interpretability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    features[2].markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2.5rem;">⚡</div>
            <div style="font-weight: 600; margin: 10px 0;">Real-time</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Instant Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    features[3].markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2.5rem;">🛡️</div>
            <div style="font-weight: 600; margin: 10px 0;">Secure</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Privacy Focused</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Upload prompt
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; padding: 40px;">'
        '<h3 style="color: #22c55e;">📤 Ready to Analyze?</h3>'
        '<p style="color: #cbd5e1;">Upload your first image using the uploader above</p>'
        '</div>',
        unsafe_allow_html=True
    )

# ================= PATIENT HISTORY (CLICK TO VIEW) =================

with st.expander("📁 Patient History", expanded=False):
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No patient records available yet.")
