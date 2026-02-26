import streamlit as st
from PIL import Image
import io
import json
import numpy as np
import pandas as pd
import altair as alt
import os
from retina_risk.model import predict_risk, has_trained_model, set_use_trained
from retina_risk.utils import load_image_bytes, preprocess_image, compute_vesselness, compute_stats, overlay_vesselness
from retina_risk.train import train_from_uploads, validate_from_uploads
import requests
from retina_risk.auth import authenticate_user, register_user, users_exist, ensure_storage, load_session, save_session, clear_session, seed_from_bootstrap_file
 
# Static assets
SIDEBAR_IMAGE_URL = "https://images.unsplash.com/photo-1580281657527-47a446a0e6b2?auto=format&fit=crop&w=800&q=60"

st.set_page_config(page_title="Heart Attack Risk from Retinal Image", page_icon="🩺", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"
if "use_trained" not in st.session_state:
    st.session_state.use_trained = False
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if not st.session_state.authenticated and st.session_state.page not in ("login", "signup"):
    st.session_state.page = "login"
ensure_storage()
seed_from_bootstrap_file(remove_after=True)
sess = load_session()
if sess and int(sess.get("expires", 0)) > int(__import__("time").time()):
    st.session_state.authenticated = True
    st.session_state.user = sess.get("user", "")
    st.session_state.page = "home"

st.markdown(
    """
    <style>
    :root {
      --primary:#e91e63;
      --accent:#00bcd4;
      --bg-grad: linear-gradient(120deg, rgba(233,30,99,0.12), rgba(0,188,212,0.12));
      --card-bg: #0f172a;
      --card-border: #334155;
      --good:#22c55e;
      --mid:#f59e0b;
      --bad:#ef4444;
    }
    .app-banner {
      background: var(--bg-grad);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 22px;
      margin-bottom: 12px;
    }
    .app-title {
      font-size: 28px;
      font-weight: 700;
      color: #e2e8f0;
      letter-spacing: .5px;
    }
    .app-sub {
      color: #94a3b8;
      margin-top: 4px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 12px;
    }
    .risk-value {
      font-size: 36px;
      font-weight: 800;
      margin-top: 8px;
    }
    .risk-chip {
      display:inline-block;
      padding:6px 10px;
      border-radius:999px;
      font-weight:600;
      color:#0b0f19;
      margin-left:8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def _fetch_image_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=6)
    r.raise_for_status()
    return r.content

if st.session_state.page == "home":
    with st.sidebar:
        try:
            b = _fetch_image_bytes(SIDEBAR_IMAGE_URL)
            st.image(Image.open(io.BytesIO(b)), use_column_width=True)
        except Exception:
            st.image("https://static.streamlit.io/examples/dice.jpg", use_column_width=True)
        st.markdown("<div class='card'><b>About</b><br>Retinal vasculature features are correlated with cardiovascular risk. This app visualizes vessels and computes a heuristic risk score.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><b>Legend</b><br><span style='color:var(--good)'>Low</span> < 25 • <span style='color:var(--mid)'>Moderate</span> 25–50 • <span style='color:var(--accent)'>Elevated</span> 50–75 • <span style='color:var(--bad)'>High</span> ≥ 75</div>", unsafe_allow_html=True)
        if has_trained_model():
            st.toggle("Use trained model", value=st.session_state.use_trained, key="toggle_use_trained")
            set_use_trained(st.session_state.toggle_use_trained)
            st.session_state.use_trained = st.session_state.toggle_use_trained
        if st.session_state.authenticated:
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.user = ""
                st.session_state.page = "login"
                clear_session()
                st.rerun()

    st.markdown("<div class='app-banner'><div class='app-title'>🩺 Heart Attack Risk Prediction</div><div class='app-sub'>Upload a retinal fundus image to analyze vasculature and estimate risk</div></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded:
        data = uploaded.read()
        img = load_image_bytes(data)
        pre = preprocess_image(img)
        vess = compute_vesselness(pre)
        stats = compute_stats(pre, vess)
        score = predict_risk(stats)

        level = "Low"
        color = "var(--good)"
        if score >= 75:
            level = "High"
            color = "var(--bad)"
        elif score >= 50:
            level = "Elevated"
            color = "var(--accent)"
        elif score >= 25:
            level = "Moderate"
            color = "var(--mid)"

        left, right = st.columns([1.2, 1])
        with left:
            img_tabs = st.tabs(["Original", "Vesselness Overlay"])
            with img_tabs[0]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.image(Image.open(io.BytesIO(data)), use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with img_tabs[1]:
                ov = overlay_vesselness(pre, vess)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.image(ov, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Risk Summary")
            st.markdown(f"<div class='risk-value' style='color:{color}'>{score:.1f} / 100 <span class='risk-chip' style='background:{color}'>{level}</span></div>", unsafe_allow_html=True)
            st.progress(min(max(score / 100.0, 0.0), 1.0))
            st.markdown("</div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("<div class='card'>Intensity mean</div>", unsafe_allow_html=True)
            st.metric(label="", value=f"{stats['intensity_mean']:.3f}")
        with m2:
            st.markdown("<div class='card'>Intensity std</div>", unsafe_allow_html=True)
            st.metric(label="", value=f"{stats['intensity_std']:.3f}")
        with m3:
            st.markdown("<div class='card'>Vesselness mean</div>", unsafe_allow_html=True)
            st.metric(label="", value=f"{stats['vesselness_mean']:.3f}")
        with m4:
            st.markdown("<div class='card'>Vesselness std</div>", unsafe_allow_html=True)
            st.metric(label="", value=f"{stats['vesselness_std']:.3f}")

        tabs = st.tabs(["Analysis", "Report", "Details"])
        with tabs[0]:
            df_int = pd.DataFrame({"value": pre.flatten()})
            df_ves = pd.DataFrame({"value": vess.flatten()})
            chart_int = alt.Chart(df_int).mark_area(
                line={"color": "#e91e63"},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[{"color": "#e91e63", "offset": 0}, {"color": "#00bcd4", "offset": 1}],
                    x1=1, x2=0, y1=1, y2=0,
                ),
                opacity=0.6,
            ).encode(
                alt.X("value:Q", bin=alt.Bin(maxbins=30), title="Intensity"),
                alt.Y("count()", title="Count"),
            ).properties(title="Intensity Distribution", height=220)

            chart_ves = alt.Chart(df_ves).mark_area(
                line={"color": "#00bcd4"},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[{"color": "#00bcd4", "offset": 0}, {"color": "#e91e63", "offset": 1}],
                    x1=0, x2=1, y1=0, y2=1,
                ),
                opacity=0.6,
            ).encode(
                alt.X("value:Q", bin=alt.Bin(maxbins=30), title="Vesselness"),
                alt.Y("count()", title="Count"),
            ).properties(title="Vesselness Distribution", height=220)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.altair_chart(chart_int, use_container_width=True)
            st.altair_chart(chart_ves, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        report = {
            "filename": uploaded.name,
            "intensity_mean": stats["intensity_mean"],
            "intensity_std": stats["intensity_std"],
            "vesselness_mean": stats["vesselness_mean"],
            "vesselness_std": stats["vesselness_std"],
            "risk_score": score,
            "risk_level": level,
        }
        with tabs[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Report")
            st.json(report)
            st.download_button("Download report (JSON)", data=json.dumps(report, indent=2), file_name="retina_risk_report.json", mime="application/json")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><b>Medical accuracy</b> → <span style='color:var(--mid)'>Needs validation</span><br>Open validation to evaluate metrics on labeled data.</div>", unsafe_allow_html=True)
            if st.button("Open validation", key="btn_report_validate"):
                st.session_state.page = "accuracy"
                st.rerun()
        with tabs[2]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Detailed Analysis (Simple English)")
            bright = "moderate"
            if stats["intensity_mean"] < 0.3:
                bright = "low (darker image)"
            elif stats["intensity_mean"] > 0.7:
                bright = "high (brighter image)"
            vess_level = "moderate"
            if stats["vesselness_mean"] < 0.1:
                vess_level = "faint vessels"
            elif stats["vesselness_mean"] > 0.3:
                vess_level = "pronounced vessels"
            vess_var = "typical variation"
            if stats["vesselness_std"] < 0.05:
                vess_var = "very uniform"
            elif stats["vesselness_std"] > 0.15:
                vess_var = "high variation"
            simple = (
                f"Risk score is {score:.1f}/100, which falls in the {level} range. "
                f"The image brightness appears {bright}. "
                f"Blood vessel visibility is {vess_level}, with {vess_var} across the image. "
                "These patterns can reflect how clearly vessels are captured and how much they differ across regions."
            )
            st.write(simple)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("What this means:")
            st.markdown(f"- A {level.lower()} score suggests {'lower' if level=='Low' else 'higher' if level=='High' else 'some'} likelihood of cardiovascular risk based on image features.")
            st.markdown("- Brightness and vessel clarity can affect feature measurements; consistent lighting improves reliability.")
            st.markdown("- This is not a medical diagnosis. For reliable medical use, validate with labeled data in the Validation page.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><b>Explore More</b><br>Select a topic to see full details.</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='card'>Used Technologies<br><span style='color:#94a3b8'>Streamlit, NumPy, Pandas, scikit-image, Pillow, Altair</span></div>", unsafe_allow_html=True)
        if st.button("View details", key="btn_tech"):
            st.session_state.page = "tech"
            st.rerun()
    with c2:
        st.markdown("<div class='card'>ML Algorithm<br><span style='color:#94a3b8'>Heuristic risk mapping from image features</span></div>", unsafe_allow_html=True)
        if st.button("View details", key="btn_ml"):
            st.session_state.page = "ml"
            st.rerun()
    with c3:
        st.markdown("<div class='card'>Image Processing<br><span style='color:#94a3b8'>Grayscale normalization and Frangi vesselness</span></div>", unsafe_allow_html=True)
        if st.button("View details", key="btn_vision"):
            st.session_state.page = "vision"
            st.rerun()
    with c4:
        st.markdown("<div class='card'>App Framework<br><span style='color:#94a3b8'>Streamlit UI, tabs, charts, report system</span></div>", unsafe_allow_html=True)
        if st.button("View details", key="btn_framework"):
            st.session_state.page = "framework"
            st.rerun()

elif st.session_state.page == "login":
    st.markdown(
        """
        <style>
        .stForm {
          background:#ffffff !important;
          border:1px solid #e2e8f0 !important;
          border-radius:14px !important;
          padding:22px !important;
          box-shadow:0 6px 24px rgba(0,0,0,0.12) !important;
        }
        .login-header {
          max-width: 420px;
          margin: 0 auto 12px auto;
          background:#ffffff;
          border:1px solid #e2e8f0;
          border-radius:14px;
          padding:16px;
          box-shadow:0 6px 24px rgba(0,0,0,0.12);
          text-align:left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="login-header" style="margin-top:24px;">
          <div style="color:#64748b;font-weight:600;">Please enter your details</div>
          <div style="font-size:30px;font-weight:800;color:#0b0f19;margin:6px 0 4px;">Welcome back</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("login_form", clear_on_submit=False):
        lu = st.text_input("Email address", placeholder="you@example.com", label_visibility="collapsed")
        lp = st.text_input("Password", type="password", placeholder="••••••••", label_visibility="collapsed")
        c1, c2 = st.columns([1,1])
        with c1:
            remember = st.checkbox("Remember for 30 days")
        with c2:
            forgot = st.form_submit_button("Forgot password")
        submit = st.form_submit_button("Sign in")
        if forgot:
            st.info("Local app: reset by creating a new account or removing storage/users.json")
        if submit:
            if authenticate_user(lu, lp):
                st.session_state.authenticated = True
                st.session_state.user = lu
                if remember:
                    save_session(lu, 30)
                st.session_state.page = "home"
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.markdown("<div style='text-align:center;color:#64748b;margin-top:8px;'>Don't have an account?</div>", unsafe_allow_html=True)
    if st.button("Go to Sign up"):
        st.session_state.page = "signup"
        st.rerun()
elif st.session_state.page == "signup":
    st.markdown(
        """
        <style>
        .stForm {
          background:#ffffff !important;
          border:1px solid #e2e8f0 !important;
          border-radius:14px !important;
          padding:22px !important;
          box-shadow:0 6px 24px rgba(0,0,0,0.12) !important;
        }
        .login-header {
          max-width: 420px;
          margin: 0 auto 12px auto;
          background:#ffffff;
          border:1px solid #e2e8f0;
          border-radius:14px;
          padding:16px;
          box-shadow:0 6px 24px rgba(0,0,0,0.12);
          text-align:left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="login-header" style="margin-top:24px;">
          <div style="color:#64748b;font-weight:600;">Create your account</div>
          <div style="font-size:30px;font-weight:800;color:#0b0f19;margin:6px 0 4px;">Sign up</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("signup_form", clear_on_submit=False):
        su_name = st.text_input("Email address", placeholder="you@example.com", label_visibility="collapsed")
        su_pass = st.text_input("Password", type="password", placeholder="Choose a strong password", label_visibility="collapsed")
        create = st.form_submit_button("Create account")
        if create:
            ok = register_user(su_name, su_pass)
            if ok:
                st.success("Account created. You can sign in now.")
            else:
                st.error("Sign up failed. Try a different email and non-empty password.")
    st.markdown("<div style='text-align:center;color:#64748b;margin-top:8px;'>Already have an account?</div>", unsafe_allow_html=True)
    if st.button("Go to Sign in"):
        st.session_state.page = "login"
        st.rerun()
else:
    st.markdown("<div class='app-banner'><div class='app-title'>📘 Details</div><div class='app-sub'>Expanded information for the selected topic</div></div>", unsafe_allow_html=True)
    back = st.button("← Back to Analysis")
    if back:
        st.session_state.page = "home"
        st.rerun()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.session_state.page == "tech":
        st.subheader("Used Technologies")
        st.markdown("- Streamlit: UI rendering, layout, widgets, charts")
        st.markdown("- NumPy: numerical arrays and statistics")
        st.markdown("- Pandas: table formation for charting")
        st.markdown("- scikit-image: Frangi vesselness, resizing, grayscale conversion")
        st.markdown("- Pillow: image decoding and conversion")
        st.markdown("- Altair: declarative charts for distributions")
    elif st.session_state.page == "ml":
        st.subheader("ML Algorithm")
        st.markdown("- Inputs: intensity mean/std, vesselness mean/std")
        st.markdown("- Mapping: logistic-style heuristic producing 0–100 score")
        st.markdown("- Rationale: higher vesselness and variability may correlate with vascular changes")
        st.markdown("- Note: not a clinically validated model; replaceable with trained classifier")
    elif st.session_state.page == "vision":
        st.subheader("Image Processing")
        st.markdown("- Preprocessing: proportional resize, grayscale normalization")
        st.markdown("- Vesselness: Frangi filter highlighting tubular structures")
        st.markdown("- Overlay: vesselness blended into blue channel for visualization")
        st.markdown("- Metrics: feature distributions presented with interactive charts")
    elif st.session_state.page == "framework":
        st.subheader("App Framework")
        st.markdown("- Layout: wide view, cards, tabs for analysis/report")
        st.markdown("- Navigation: session-state based detail views with back action")
        st.markdown("- Portability: requirements file and PowerShell bootstrap script")
        st.markdown("- Reporting: downloadable JSON with metrics and risk")
    elif st.session_state.page == "train":
        st.subheader("Train Model")
        st.markdown("Upload a labels CSV and multiple images. CSV must have columns: filename,label where label ∈ {0,1}. Filenames should match uploaded image names.")
        labels = st.file_uploader("Labels CSV", type=["csv"], key="labels_csv")
        imgs = st.file_uploader("Training Images", type=["jpg","jpeg","png","bmp"], accept_multiple_files=True, key="train_images")
        if labels and imgs:
            save_path = os.path.join(os.path.dirname(__file__), "retina_risk", "model.pkl")
            res = train_from_uploads(imgs, labels.read(), save_path)
            if res["success"]:
                st.success(f"Training completed on {res['count']} samples. Accuracy: {res['accuracy']:.3f}")
                if "cv" in res:
                    cv = res["cv"]
                    st.info(f"Cross-validation: Acc {cv['accuracy']:.3f} • Prec {cv['precision']:.3f} • Rec {cv['recall']:.3f} • F1 {cv['f1']:.3f} • AUC {cv['auc']:.3f}")
                    roc_df = pd.DataFrame({"fpr": cv["roc_curve"]["fpr"], "tpr": cv["roc_curve"]["tpr"]})
                    cm = np.array(cv["confusion_matrix"])
                    cm_df = pd.DataFrame({"x": ["Negative","Positive","Negative","Positive"], "y": ["Negative","Negative","Positive","Positive"], "value": cm.flatten()})
                    roc_chart = alt.Chart(roc_df).mark_line(color="#00bcd4").encode(alt.X("fpr:Q", title="False Positive Rate"), alt.Y("tpr:Q", title="True Positive Rate")).properties(title="ROC Curve", height=220)
                    cm_chart = alt.Chart(cm_df).mark_rect().encode(x="x:N", y="y:N", color=alt.Color("value:Q", scale=alt.Scale(scheme="reds"))).properties(title="Confusion Matrix", height=220)
                    st.altair_chart(roc_chart, use_container_width=True)
                    st.altair_chart(cm_chart, use_container_width=True)
                st.session_state.use_trained = True
                set_use_trained(True)
            else:
                st.error(res["message"])
    elif st.session_state.page == "accuracy":
        st.subheader("Medical Accuracy Validation")
        st.markdown("Option A: Upload a labeled validation set and evaluate the trained model.")
        val_labels = st.file_uploader("Validation Labels CSV", type=["csv"], key="val_labels_csv")
        val_imgs = st.file_uploader("Validation Images", type=["jpg","jpeg","png","bmp"], accept_multiple_files=True, key="val_images")
        if val_labels and val_imgs:
            model_path = os.path.join(os.path.dirname(__file__), "retina_risk", "model.pkl")
            res = validate_from_uploads(val_imgs, val_labels.read(), model_path)
            if res["success"]:
                st.success(f"Validation on {res['count']} samples")
                st.info(f"Accuracy {res['accuracy']:.3f} • Precision {res['precision']:.3f} • Recall {res['recall']:.3f} • F1 {res['f1']:.3f} • AUC {res['auc']:.3f}")
                roc_df = pd.DataFrame({"fpr": res["roc_curve"]["fpr"], "tpr": res["roc_curve"]["tpr"]})
                cm = np.array(res["confusion_matrix"])
                cm_df = pd.DataFrame({"x": ["Negative","Positive","Negative","Positive"], "y": ["Negative","Negative","Positive","Positive"], "value": cm.flatten()})
                roc_chart = alt.Chart(roc_df).mark_line(color="#e91e63").encode(alt.X("fpr:Q", title="False Positive Rate"), alt.Y("tpr:Q", title="True Positive Rate")).properties(title="ROC Curve (Validation)", height=220)
                cm_chart = alt.Chart(cm_df).mark_rect().encode(x="x:N", y="y:N", color=alt.Color("value:Q", scale=alt.Scale(scheme="blues"))).properties(title="Confusion Matrix (Validation)", height=220)
                st.altair_chart(roc_chart, use_container_width=True)
                st.altair_chart(cm_chart, use_container_width=True)
            else:
                st.error(res["message"])
        st.markdown("Option B: Use cross-validation during training for stability across folds.")
    st.markdown("</div>", unsafe_allow_html=True)
