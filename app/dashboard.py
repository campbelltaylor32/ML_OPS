"""
dashboard.py  --  Steps 7 & 12: deployed inference UI + monitoring dashboard
=============================================================================
MLOps purpose: DEPLOYMENT (interactive serving) + MONITORING (visual dashboard).
Two tabs:
  1. Predict        -- score a single student live (the deployed model in action)
  2. Monitoring     -- original vs modified test set: metric degradation,
                       feature drift (PSI) table, and prediction-shift chart.

This is the screen to record for the video demo (assignment step 11).

Run:  streamlit run app/dashboard.py
"""
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.evaluate import regression_metrics
from src.monitoring import compute_drift

st.set_page_config(page_title="Student Performance Risk Monitor", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PATH)


@st.cache_data
def load_data():
    orig = pd.read_csv(config.TEST_PATH)
    mod = (pd.read_csv(config.TEST_MODIFIED_PATH)
           if config.TEST_MODIFIED_PATH.exists() else None)
    return orig, mod


model = load_model()
meta = json.loads(config.MODEL_META_PATH.read_text())
orig_df, mod_df = load_data()

st.title("🎓 Student Performance Risk Monitoring System")
st.caption(f"Deployed model: **{meta['best_algorithm']}**  |  "
           f"Target: **{config.TARGET}**  |  "
           f"At-Risk if predicted score < {config.RISK_THRESHOLD:.0f}")

tab_predict, tab_monitor = st.tabs(["🔮 Predict", "📡 Monitoring"])

# --------------------------------------------------------------------------- #
# TAB 1 -- live single-student inference
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Score an individual student")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 25, 21)
        study = st.slider("Study Hours / day", 0.0, 15.0, 4.0, 0.5)
        sleep = st.slider("Sleep Hours / day", 2.0, 11.0, 7.0, 0.5)
        attend = st.slider("Attendance Rate", 40.0, 100.0, 80.0, 1.0)
        stress = st.slider("Stress Level", 1.0, 10.0, 4.0, 0.1)
    with c2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        major = st.selectbox("Major", ["STEM", "Arts", "Business", "Humanities"])
        living = st.selectbox("Living Area", ["Urban", "Suburban", "Rural"])
        parent = st.selectbox("Parent Education",
                              ["High School", "Bachelor", "Master", "PhD"])
    with c3:
        scholarship = st.selectbox("Scholarship", ["No", "Yes"])
        job = st.selectbox("Part-time Job", ["No", "Yes"])
        extra = st.selectbox("Extracurriculars", ["No", "Yes"])
        internet = st.selectbox("Internet Access", ["Yes", "No"])
        exam = st.selectbox("Exam Proximity", ["No", "Yes"])

    row = pd.DataFrame([{
        "Age": age, "Study_Hours_Per_Day": study, "Sleep_Hours_Per_Day": sleep,
        "Attendance_Rate": attend, "Stress_Level": stress, "Gender": gender,
        "Major": major, "Living_Area": living, "Parent_Education": parent,
        "Scholarship_Status": scholarship, "Part_Time_Job": job,
        "Extracurricular_Activities": extra, "Internet_Access": internet,
        "Exam_Proximity": exam,
    }])[config.FEATURES]

    score = float(model.predict(row)[0])
    at_risk = score < config.RISK_THRESHOLD
    m1, m2 = st.columns(2)
    m1.metric("Predicted Performance Score", f"{score:.1f}")
    m2.metric("Status", "⚠️ AT RISK" if at_risk else "✅ Not at risk")

# --------------------------------------------------------------------------- #
# TAB 2 -- monitoring dashboard
# --------------------------------------------------------------------------- #
with tab_monitor:
    st.subheader("Original vs Modified ('academic stress') test set")
    if mod_df is None:
        st.warning("Run `python -m src.make_modified_test` first to generate "
                   "the modified test set.")
    else:
        orig_pred = model.predict(orig_df[config.FEATURES])
        mod_pred = model.predict(mod_df[config.FEATURES])
        om = regression_metrics(orig_df[config.TARGET], orig_pred)
        mm = regression_metrics(mod_df[config.TARGET], mod_pred)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("RMSE", f"{mm['RMSE']:.2f}", f"{mm['RMSE']-om['RMSE']:+.2f}",
                  delta_color="inverse")
        k2.metric("MAE", f"{mm['MAE']:.2f}", f"{mm['MAE']-om['MAE']:+.2f}",
                  delta_color="inverse")
        k3.metric("R²", f"{mm['R2']:.2f}", f"{mm['R2']-om['R2']:+.2f}")
        k4.metric("Mean predicted score",
                  f"{mod_pred.mean():.1f}",
                  f"{mod_pred.mean()-orig_pred.mean():+.1f}", delta_color="inverse")

        r1 = (orig_pred < config.RISK_THRESHOLD).mean() * 100
        r2 = (mod_pred < config.RISK_THRESHOLD).mean() * 100
        st.metric("Students flagged At-Risk", f"{r2:.0f}%", f"{r2-r1:+.0f} pts",
                  delta_color="inverse")

        # Prediction distribution overlay
        fig = go.Figure()
        fig.add_histogram(x=orig_pred, name="Original", opacity=0.6, nbinsx=25)
        fig.add_histogram(x=mod_pred, name="Modified (stress)", opacity=0.6, nbinsx=25)
        fig.add_vline(x=config.RISK_THRESHOLD, line_dash="dash",
                      annotation_text="At-Risk threshold")
        fig.update_layout(barmode="overlay", title="Predicted score distribution",
                          xaxis_title="Predicted score", yaxis_title="Students")
        st.plotly_chart(fig, use_container_width=True)

        # Drift (PSI) table
        st.subheader("Feature & prediction drift (PSI)")
        report = compute_drift(orig_df, mod_df, model)
        drift_rows = [{"feature": k, "PSI": v["psi"], "status": v["status"]}
                      for k, v in report["features"].items()]
        drift_rows.append({"feature": "PREDICTION", "PSI": report["prediction"]["psi"],
                           "status": report["prediction"]["status"]})
        drift_df = pd.DataFrame(drift_rows).sort_values("PSI", ascending=False)

        def color(val):
            if val >= 0.25:
                return "background-color: #f8d7da"
            if val >= 0.10:
                return "background-color: #fff3cd"
            return "background-color: #d4edda"

        st.dataframe(drift_df.style.applymap(color, subset=["PSI"]),
                     use_container_width=True, hide_index=True)
        st.info(f"Dataset drift detected: **{report['summary']['dataset_drift']}** "
                f"({report['summary']['n_drifted']} of "
                f"{report['summary']['n_features']} features drifted). "
                "PSI < 0.10 none · 0.10–0.25 moderate · > 0.25 major.")
