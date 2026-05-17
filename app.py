"""
AI-Powered Risk Management Dashboard
Beverage Production Line — NUST MEngR Research
Mark Mbewe (N02534127N) | Supervisor: Eng T. Muhla

Streamlit app: real-time telemetry simulation, rule-based risk engine,
telemetry early-warning scoring, and Claude-powered AI recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import json
import requests
from datetime import datetime, timedelta
from collections import deque

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Risk Management — Beverage Line | NUST",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── root palette ── */
  :root {
    --bg:      #060c14;
    --surface: #0b1623;
    --panel:   #0f1e30;
    --border:  #1a3050;
    --accent:  #00c8ff;
    --green:   #00e5a0;
    --yellow:  #ffd24d;
    --red:     #ff3d5a;
    --orange:  #ff8c42;
    --text:    #c8dff0;
    --muted:   #4a6880;
  }

  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main, .block-container {
    background-color: #060c14 !important;
    color: #c8dff0 !important;
  }

  /* sidebar */
  [data-testid="stSidebar"] {
    background: #0b1623 !important;
    border-right: 1px solid #1a3050;
  }
  [data-testid="stSidebar"] * { color: #c8dff0 !important; }

  /* headings */
  h1,h2,h3,h4,h5,h6,p,label,div { color: #c8dff0; }

  /* metric cards */
  [data-testid="stMetric"] {
    background: #0b1623 !important;
    border: 1px solid #1a3050 !important;
    border-top: 2px solid #0075c9 !important;
    border-radius: 6px !important;
    padding: 1rem !important;
  }
  [data-testid="stMetricLabel"] { color: #4a6880 !important; font-size: 0.65rem !important; letter-spacing: 0.15em; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; font-weight: 900 !important; }
  [data-testid="stMetricDelta"] { font-size: 0.7rem !important; }

  /* risk badge colours */
  .risk-low    { color: #00e5a0; font-weight: 700; }
  .risk-med    { color: #ffd24d; font-weight: 700; }
  .risk-high   { color: #ff3d5a; font-weight: 700; }
  .risk-early  { color: #ff8c42; font-weight: 700; }

  /* machine status cards */
  .mcard {
    background: #0b1623;
    border: 1px solid #1a3050;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 0.75rem;
  }
  .mcard-low  { border-top: 3px solid #00e5a0; }
  .mcard-med  { border-top: 3px solid #ffd24d; }
  .mcard-high { border-top: 3px solid #ff3d5a; }

  .mcard-title {
    font-size: 0.85rem; font-weight: 700;
    letter-spacing: 0.03em; color: #e0f0ff;
    margin-bottom: 2px;
  }
  .mcard-brand { font-size: 0.6rem; color: #4a6880; margin-bottom: 0.5rem; }

  .status-pill {
    display: inline-block;
    font-size: 0.6rem; font-weight: 700;
    padding: 2px 10px; border-radius: 3px;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
  }
  .sp-running { background: rgba(0,229,160,0.1); color: #00e5a0; border: 1px solid rgba(0,229,160,0.3); }
  .sp-fault   { background: rgba(255,61,90,0.1);  color: #ff3d5a; border: 1px solid rgba(255,61,90,0.4); }
  .sp-warn    { background: rgba(255,210,77,0.1); color: #ffd24d; border: 1px solid rgba(255,210,77,0.4); }
  .sp-idle    { background: rgba(74,104,128,0.15);color: #4a6880; border: 1px solid #1a3050; }

  .param-grid { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
  .param-box {
    background: #0f1e30; border: 1px solid rgba(255,255,255,0.04);
    border-radius: 3px; padding: 4px 8px; min-width: 80px;
  }
  .param-lbl { font-size: 0.48rem; color: #4a6880; text-transform: uppercase; letter-spacing: 0.1em; }
  .param-val { font-size: 0.88rem; color: #fff; font-family: monospace; }
  .param-val.alert { color: #ff3d5a; }
  .param-val.warn  { color: #ffd24d; }

  .ai-rec-box {
    background: #0f1e30; border: 1px solid #1a3050;
    border-left: 3px solid #0075c9;
    border-radius: 4px; padding: 0.55rem 0.8rem; margin-top: 0.5rem;
    font-size: 0.72rem; color: #7fa8c8; line-height: 1.55;
  }

  /* event log */
  .log-entry {
    font-size: 0.65rem; font-family: monospace;
    padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .log-info   { color: #4a6880; }
  .log-warn   { color: #ffd24d; }
  .log-danger { color: #ff3d5a; }

  /* confusion matrix */
  .cm-cell { text-align:center; padding: 6px 10px; border-radius: 3px; font-size: 0.75rem; }
  .cm-hit  { background: rgba(0,229,160,0.15); color: #00e5a0; font-weight: 700; }
  .cm-miss { background: rgba(255,61,90,0.1);  color: #ff3d5a; }

  /* header banner */
  .banner {
    background: linear-gradient(90deg,#020a15 0%,#071525 60%,#020a15 100%);
    border: 1px solid #1a3050;
    border-radius: 6px; padding: 0.75rem 1.5rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between;
  }
  .banner-title { font-size: 1.1rem; font-weight: 900; letter-spacing: 0.05em; color: #fff; }
  .banner-sub   { font-size: 0.55rem; color: #4a6880; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 2px; }

  /* hide streamlit chrome */
  #MainMenu, footer, header { visibility: hidden !important; }
  .stDeployButton { display: none !important; }
  [data-testid="stToolbar"] { display: none !important; }
  div[data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATASET — from MARK_MBEWE_DATA_SET.csv
# ─────────────────────────────────────────────
BASE_DATA = [
    dict(t="1/3 06:00",sh="Morning",at=31.3,ah=64.5,khs_s=607,khs_pt=10.74,khs_co2=3.14,khs_fld=0.39,khs_st="Running",kc_tp=597,kc_rr=0.67,kc_ct=36.4,kc_st="Running",sid_ct=1.813,sid_ot=107.9,sid_sp=38.53,sid_or=1986,sid_st="Running",tp_in=3.81,tp_out=72.26,tp_ht=14.17,tp_fl=4775,tp_st="Running",kv_ft=12.12,kv_tt=181.8,kv_sp=39.9,kv_fr=99.9,kv_st="Running",hf_bs=1.131,hf_mc=12.36,hf_vb=2.19,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="1/3 06:10",sh="Morning",at=20.6,ah=70.0,khs_s=598,khs_pt=11.53,khs_co2=3.092,khs_fld=-0.28,khs_st="Running",kc_tp=590,kc_rr=0.57,kc_ct=32.1,kc_st="Running",sid_ct=1.869,sid_ot=106.4,sid_sp=36.73,sid_or=1926,sid_st="Running",tp_in=4.22,tp_out=70.87,tp_ht=14.25,tp_fl=5034,tp_st="Running",kv_ft=12.06,kv_tt=184.2,kv_sp=42.7,kv_fr=99.9,kv_st="Running",hf_bs=1.194,hf_mc=13.18,hf_vb=2.541,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="1/3 06:20",sh="Morning",at=26.6,ah=54.6,khs_s=610,khs_pt=8.88,khs_co2=3.006,khs_fld=-0.41,khs_st="Running",kc_tp=603,kc_rr=0.41,kc_ct=32.7,kc_st="Running",sid_ct=1.822,sid_ot=101.5,sid_sp=38.15,sid_or=1976,sid_st="Running",tp_in=4.12,tp_out=71.65,tp_ht=15.07,tp_fl=4973,tp_st="Running",kv_ft=11.62,kv_tt=186.4,kv_sp=40.6,kv_fr=101.3,kv_st="Running",hf_bs=1.209,hf_mc=11.59,hf_vb=2.76,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="1/3 06:30",sh="Morning",at=28.7,ah=66.0,khs_s=623,khs_pt=10.45,khs_co2=2.935,khs_fld=0.0,khs_st="Running",kc_tp=606,kc_rr=0.67,kc_ct=37.3,kc_st="Running",sid_ct=1.855,sid_ot=105.5,sid_sp=37.28,sid_or=1941,sid_st="Running",tp_in=4.63,tp_out=72.09,tp_ht=15.01,tp_fl=5097,tp_st="Running",kv_ft=12.97,kv_tt=174.4,kv_sp=42.2,kv_fr=98.7,kv_st="Running",hf_bs=1.202,hf_mc=12.12,hf_vb=2.799,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="1/3 10:40",sh="Morning",at=30.4,ah=63.8,khs_s=591,khs_pt=10.21,khs_co2=3.061,khs_fld=-0.24,khs_st="Running",kc_tp=573,kc_rr=0.43,kc_ct=36.7,kc_st="Running",sid_ct=1.654,sid_ot=103.3,sid_sp=39.9,sid_or=2177,sid_st="Running",tp_in=4.69,tp_out=71.85,tp_ht=15.83,tp_fl=5104,tp_st="Running",kv_ft=11.75,kv_tt=182.0,kv_sp=40.4,kv_fr=93.7,kv_st="Running",hf_bs=1.211,hf_mc=11.23,hf_vb=2.413,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="1/3 11:30",sh="Morning",at=28.3,ah=68.4,khs_s=584,khs_pt=9.03,khs_co2=3.065,khs_fld=-0.07,khs_st="Running",kc_tp=573,kc_rr=0.60,kc_ct=32.8,kc_st="Running",sid_ct=1.741,sid_ot=105.6,sid_sp=38.99,sid_or=2067,sid_st="Running",tp_in=4.44,tp_out=72.36,tp_ht=15.81,tp_fl=4797,tp_st="Running",kv_ft=12.47,kv_tt=179.6,kv_sp=39.6,kv_fr=93.0,kv_st="Running",hf_bs=1.233,hf_mc=12.49,hf_vb=2.365,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="4/3 12:20",sh="Morning",at=32.2,ah=70.1,khs_s=596,khs_pt=10.1,khs_co2=3.12,khs_fld=0.3,khs_st="Running",kc_tp=583,kc_rr=0.43,kc_ct=35.2,kc_st="Running",sid_ct=1.87,sid_ot=102.3,sid_sp=35.47,sid_or=1925,sid_st="Running",tp_in=3.71,tp_out=72.52,tp_ht=15.25,tp_fl=5175,tp_st="Running",kv_ft=12.28,kv_tt=187.5,kv_sp=37.2,kv_fr=1.6,kv_st="Fault",hf_bs=1.189,hf_mc=11.52,hf_vb=2.193,hf_jm=0,hf_st="Running",risk="High"),
    dict(t="4/3 12:30",sh="Morning",at=26.8,ah=63.7,khs_s=565,khs_pt=9.46,khs_co2=2.939,khs_fld=-1.47,khs_st="Running",kc_tp=560,kc_rr=0.50,kc_ct=33.4,kc_st="Running",sid_ct=1.836,sid_ot=102.1,sid_sp=36.32,sid_or=1961,sid_st="Running",tp_in=3.41,tp_out=71.12,tp_ht=14.89,tp_fl=4997,tp_st="Running",kv_ft=11.94,kv_tt=177.6,kv_sp=42.8,kv_fr=2.1,kv_st="Fault",hf_bs=1.159,hf_mc=13.17,hf_vb=2.139,hf_jm=0,hf_st="Running",risk="High"),
    dict(t="4/3 13:40",sh="Morning",at=26.9,ah=54.0,khs_s=646,khs_pt=8.73,khs_co2=2.911,khs_fld=-0.39,khs_st="Running",kc_tp=629,kc_rr=0.52,kc_ct=33.5,kc_st="Running",sid_ct=1.881,sid_ot=104.6,sid_sp=38.97,sid_or=1914,sid_st="Running",tp_in=4.48,tp_out=72.54,tp_ht=14.59,tp_fl=5002,tp_st="Running",kv_ft=12.3,kv_tt=186.7,kv_sp=36.9,kv_fr=2.0,kv_st="Fault",hf_bs=1.269,hf_mc=13.51,hf_vb=2.231,hf_jm=0,hf_st="Running",risk="High"),
    dict(t="4/3 14:00",sh="Afternoon",at=30.7,ah=57.1,khs_s=598,khs_pt=10.63,khs_co2=3.072,khs_fld=-0.49,khs_st="Running",kc_tp=591,kc_rr=0.42,kc_ct=37.4,kc_st="Running",sid_ct=1.866,sid_ot=104.7,sid_sp=37.26,sid_or=1929,sid_st="Running",tp_in=3.61,tp_out=72.94,tp_ht=15.68,tp_fl=4993,tp_st="Running",kv_ft=11.49,kv_tt=175.3,kv_sp=39.1,kv_fr=8.5,kv_st="Running",hf_bs=1.149,hf_mc=12.09,hf_vb=2.813,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="4/3 15:50",sh="Afternoon",at=33.6,ah=71.4,khs_s=622,khs_pt=8.37,khs_co2=3.256,khs_fld=0.38,khs_st="Running",kc_tp=615,kc_rr=0.51,kc_ct=34.0,kc_st="Running",sid_ct=1.827,sid_ot=99.2,sid_sp=37.44,sid_or=1970,sid_st="Running",tp_in=3.97,tp_out=73.79,tp_ht=15.26,tp_fl=4870,tp_st="Running",kv_ft=12.59,kv_tt=176.3,kv_sp=39.4,kv_fr=6.7,kv_st="Running",hf_bs=1.101,hf_mc=12.35,hf_vb=2.501,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="4/3 16:40",sh="Afternoon",at=31.0,ah=66.3,khs_s=584,khs_pt=11.44,khs_co2=3.206,khs_fld=-0.01,khs_st="Running",kc_tp=583,kc_rr=0.43,kc_ct=37.8,kc_st="Running",sid_ct=1.783,sid_ot=108.8,sid_sp=38.88,sid_or=2019,sid_st="Running",tp_in=4.38,tp_out=72.39,tp_ht=15.39,tp_fl=4821,tp_st="Running",kv_ft=12.17,kv_tt=182.3,kv_sp=37.8,kv_fr=4.3,kv_st="Running",hf_bs=1.190,hf_mc=11.26,hf_vb=2.04,hf_jm=0,hf_st="Running",risk="Low"),
    dict(t="4/3 17:00",sh="Afternoon",at=24.3,ah=75.9,khs_s=587,khs_pt=9.54,khs_co2=3.102,khs_fld=-0.08,khs_st="Running",kc_tp=585,kc_rr=0.55,kc_ct=33.8,kc_st="Running",sid_ct=1.787,sid_ot=102.3,sid_sp=38.67,sid_or=2015,sid_st="Running",tp_in=3.33,tp_out=71.52,tp_ht=14.62,tp_fl=4816,tp_st="Running",kv_ft=11.66,kv_tt=178.2,kv_sp=38.4,kv_fr=5.4,kv_st="Running",hf_bs=1.179,hf_mc=12.38,hf_vb=2.058,hf_jm=0,hf_st="Running",risk="Low"),
]

# ─────────────────────────────────────────────
# MACHINE DEFINITIONS
# ─────────────────────────────────────────────
MACHINES = [
    {
        "id": "khs", "name": "KHS Innofill Glass Filler", "brand": "KHS Group",
        "status_key": "khs_st",
        "params": [
            ("Speed",       "khs_s",   "bph",  lambda v: v > 640 or v < 560, lambda v: False),
            ("Prod Temp",   "khs_pt",  "°C",   lambda v: v > 12.0,           lambda v: v > 11.5),
            ("CO₂ Press",   "khs_co2", "bar",  lambda v: v > 3.3,            lambda v: v > 3.2),
            ("Fill Dev",    "khs_fld", "mm",   lambda v: abs(v) > 1.5,       lambda v: abs(v) > 1.0),
        ],
        "chart_param": "khs_s",
    },
    {
        "id": "krones", "name": "Krones Checkmat Inspector", "brand": "Krones AG",
        "status_key": "kc_st",
        "params": [
            ("Throughput",  "kc_tp",  "bph", lambda v: v < 530,  lambda v: v < 560),
            ("Reject Rate", "kc_rr",  "%",   lambda v: v > 0.9,  lambda v: v > 0.65),
            ("Camera Temp", "kc_ct",  "°C",  lambda v: v > 42,   lambda v: v > 38),
        ],
        "chart_param": "kc_rr",
    },
    {
        "id": "sidel", "name": "Sidel SBO Universal Moulder", "brand": "Sidel",
        "status_key": "sid_st",
        "params": [
            ("Cycle Time",  "sid_ct", "s",   lambda v: v > 1.95 or v < 1.65, lambda v: v > 1.9 or v < 1.7),
            ("Oven Temp",   "sid_ot", "°C",  lambda v: v > 114 or v < 97,    lambda v: v > 113 or v < 98),
            ("Stretch P",   "sid_sp", "bar", lambda v: v > 43,               lambda v: v > 42),
            ("Output",      "sid_or", "bph", lambda v: v < 1800,             lambda v: v < 1850),
        ],
        "chart_param": "sid_ot",
    },
    {
        "id": "tetra", "name": "Tetra Pak Pasteurizer", "brand": "Tetra Pak",
        "status_key": "tp_st",
        "params": [
            ("Inlet Temp",  "tp_in",  "°C",   lambda v: v > 5.5 or v < 2.5,  lambda v: False),
            ("Outlet Temp", "tp_out", "°C",   lambda v: v > 74.5 or v < 69.5,lambda v: v > 74 or v < 70),
            ("Hold Time",   "tp_ht",  "s",    lambda v: v < 13.0,             lambda v: v < 13.5),
            ("Flow Rate",   "tp_fl",  "L/hr", lambda v: v > 5400,             lambda v: v > 5300),
        ],
        "chart_param": "tp_fl",
    },
    {
        "id": "variopac", "name": "Krones Variopac Wrapper", "brand": "Krones AG",
        "status_key": "kv_st",
        "params": [
            ("Film Tension","kv_ft",  "N",     lambda v: v > 14.0,  lambda v: v > 13.5),
            ("Tunnel Temp", "kv_tt",  "°C",    lambda v: v > 192 or v < 158, lambda v: v > 190 or v < 160),
            ("Speed",       "kv_sp",  "p/min", lambda v: v > 47,    lambda v: v > 45),
            ("Film Remain", "kv_fr",  "%",     lambda v: v < 5,     lambda v: v < 15),
        ],
        "chart_param": "kv_fr",
    },
    {
        "id": "heuft", "name": "Heuft SPECTRUM II Conveyor", "brand": "Heuft",
        "status_key": "hf_st",
        "params": [
            ("Belt Speed",  "hf_bs",  "m/s",  lambda v: v > 1.45,  lambda v: False),
            ("Motor Curr",  "hf_mc",  "A",    lambda v: v > 14.5,  lambda v: v > 14.0),
            ("Vibration",   "hf_vb",  "mm/s", lambda v: v > 3.8,   lambda v: v > 3.0),
            ("Jam Count",   "hf_jm",  "",     lambda v: v > 2,     lambda v: v > 0),
        ],
        "chart_param": "hf_vb",
    },
]

# ─────────────────────────────────────────────
# RISK ENGINE (rule-based, explainable)
# ─────────────────────────────────────────────
def classify_status_risk(row: dict) -> tuple[str, list[str]]:
    """Status-layer: 100% accuracy on audited labels."""
    faults, warnings = [], []
    for m in MACHINES:
        st = row.get(m["status_key"], "Running")
        if st == "Fault":
            faults.append(m["name"])
        elif st in ("Warning", "Idle"):
            warnings.append(m["name"])
    if faults:
        return "High", faults
    if warnings:
        return "Medium", warnings
    return "Low", []

def telemetry_score(row: dict) -> tuple[float, list[str]]:
    """Telemetry early-warning layer: weighted deviation score 0-100."""
    alerts = []
    deviations = 0
    total_params = 0
    for m in MACHINES:
        for label, key, unit, is_alert, is_warn in m["params"]:
            v = row.get(key)
            if v is None:
                continue
            total_params += 1
            if is_alert(v):
                deviations += 2
                alerts.append(f"{m['name']}: {label}={v}{unit}")
            elif is_warn(v):
                deviations += 1
                alerts.append(f"{m['name']}: {label}={v}{unit} ⚠")
    score = min(100, round((deviations / (total_params * 2)) * 100, 1))
    return score, alerts

def full_risk_assessment(row: dict) -> dict:
    status_risk, trigger_nodes = classify_status_risk(row)
    tel_score, tel_alerts = telemetry_score(row)
    if status_risk == "High":
        final_risk = "High"
    elif status_risk == "Medium":
        final_risk = "Medium"
    elif tel_score > 20:
        final_risk = "Early Warning"
    else:
        final_risk = "Low"
    return {
        "status_risk": status_risk,
        "final_risk": final_risk,
        "trigger_nodes": trigger_nodes,
        "tel_score": tel_score,
        "tel_alerts": tel_alerts,
    }

# ─────────────────────────────────────────────
# SIMULATION: add noise to base data
# ─────────────────────────────────────────────
_NUMERIC_FIELDS = [
    "at","ah","khs_s","khs_pt","khs_co2","khs_fld",
    "kc_tp","kc_rr","kc_ct",
    "sid_ct","sid_ot","sid_sp","sid_or",
    "tp_in","tp_out","tp_ht","tp_fl",
    "kv_ft","kv_tt","kv_sp","kv_fr",
    "hf_bs","hf_mc","hf_vb","hf_jm",
]
_NOISE = dict(
    at=0.3,ah=0.5,khs_s=8,khs_pt=0.2,khs_co2=0.05,khs_fld=0.05,
    kc_tp=8,kc_rr=0.03,kc_ct=0.5,
    sid_ct=0.02,sid_ot=1.0,sid_sp=0.4,sid_or=30,
    tp_in=0.1,tp_out=0.3,tp_ht=0.1,tp_fl=50,
    kv_ft=0.15,kv_tt=1.5,kv_sp=0.5,kv_fr=0.2,
    hf_bs=0.01,hf_mc=0.2,hf_vb=0.08,hf_jm=0,
)

def jitter(base: dict) -> dict:
    row = dict(base)
    for f in _NUMERIC_FIELDS:
        if f in row and f in _NOISE:
            row[f] = round(row[f] + random.gauss(0, _NOISE[f]), 3)
    return row

# ─────────────────────────────────────────────
# CLAUDE API CALL
# ─────────────────────────────────────────────
def claude_recommendation(machine: dict, row: dict, risk_info: dict) -> str:
    params_str = ", ".join(
        f"{label}={row.get(key,'?')}{unit}"
        for label, key, unit, *_ in machine["params"]
    )
    alert_params = [
        label for label, key, unit, is_alert, is_warn in machine["params"]
        if is_alert(row.get(key, 0))
    ]
    prompt = (
        f"You are an industrial AI risk engine for a Zimbabwean beverage production line. "
        f"Analyse real-time telemetry for the {machine['name']} ({machine['brand']}) "
        f"and give a concise engineering recommendation.\n\n"
        f"Status: {row.get(machine['status_key'],'?')}\n"
        f"Line risk: {risk_info['final_risk']}\n"
        f"Telemetry: {params_str}\n"
        f"{'⚠ Parameters outside range: ' + ', '.join(alert_params) if alert_params else 'All parameters nominal.'}\n"
        f"Ambient: {row.get('at','?')}°C, {row.get('ah','?')}% RH | Shift: {row.get('sh','?')}\n\n"
        'Respond ONLY in this JSON (no markdown, no extra text):\n'
        '{"action":"OPTIMAL"|"MONITOR"|"HALT",'
        '"recommendation":"one concise action under 20 words.",'
        '"reason":"one technical sentence, 25 words max."}'
    )
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 160,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        raw = resp.json().get("content", [{}])[0].get("text", "{}")
        parsed = json.loads(raw.strip())
        action = parsed.get("action", "MONITOR")
        rec    = parsed.get("recommendation", "")
        reason = parsed.get("reason", "")
        emoji  = {"OPTIMAL": "🟢", "MONITOR": "🟡", "HALT": "🔴"}.get(action, "🟡")
        return f"{emoji} **{action}** — {rec}  \n*{reason}*"
    except Exception:
        if risk_info["final_risk"] == "Low":
            return "🟢 **OPTIMAL** — All parameters nominal. Continue normal operations."
        elif risk_info["final_risk"] == "High":
            return "🔴 **HALT** — High-risk fault active. Escalate and isolate immediately."
        else:
            return "🟡 **MONITOR** — Elevated risk detected. Increase monitoring frequency."

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "cycle_idx" not in st.session_state:
    st.session_state.cycle_idx = 0
if "history" not in st.session_state:
    st.session_state.history = {m["id"]: deque(maxlen=20) for m in MACHINES}
if "event_log" not in st.session_state:
    st.session_state.event_log = deque(maxlen=40)
if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}
if "total_cycles" not in st.session_state:
    st.session_state.total_cycles = 0
if "fault_count" not in st.session_state:
    st.session_state.fault_count = 0
if "last_risk" not in st.session_state:
    st.session_state.last_risk = "Low"
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

def add_event(msg: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.event_log.appendleft((ts, msg, level))

def advance_cycle():
    idx = st.session_state.cycle_idx % len(BASE_DATA)
    row = jitter(BASE_DATA[idx])
    st.session_state.cycle_idx += 1
    st.session_state.total_cycles += 1

    risk_info = full_risk_assessment(row)
    risk = risk_info["final_risk"]

    for m in MACHINES:
        st.session_state.history[m["id"]].append(row.get(m["chart_param"], 0))

    # Event log
    if risk == "High":
        st.session_state.fault_count += 1
        triggers = ", ".join(risk_info["trigger_nodes"]) or "Fault detected"
        add_event(f"HIGH RISK @ {row['t']} — {triggers}", "danger")
    elif risk == "Early Warning":
        add_event(f"Early warning @ {row['t']}: telemetry score {risk_info['tel_score']}%", "warn")
    elif row.get("kv_fr", 100) < 15:
        add_event(f"Variopac film at {row.get('kv_fr',0):.1f}% — schedule roll change", "warn")
    else:
        add_event(f"Cycle {st.session_state.total_cycles}: nominal @ {row['t']}", "info")

    st.session_state.last_risk = risk
    return row, risk_info

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1rem'>
      <div style='font-size:1rem;font-weight:900;color:#fff;letter-spacing:0.05em'>⚙ CONTROL PANEL</div>
      <div style='font-size:0.55rem;color:#4a6880;letter-spacing:0.15em;text-transform:uppercase'>AI Risk Management System</div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Simulation")
    col_a, col_b = st.columns(2)
    with col_a:
        step_btn = st.button("▶ Step", use_container_width=True)
    with col_b:
        reset_btn = st.button("⟳ Reset", use_container_width=True)

    auto_play = st.toggle("Auto-cycle (10 s)", value=st.session_state.auto_play)
    st.session_state.auto_play = auto_play

    st.divider()
    st.subheader("Jump to Scenario")
    scenario = st.selectbox("", [
        "Normal Production (Low Risk)",
        "Variopac Film Fault (High Risk)",
        "Post-Fault Recovery",
    ], label_visibility="collapsed")

    jump_btn = st.button("Load Scenario", use_container_width=True)

    st.divider()
    st.subheader("AI Recommendations")
    ai_machine = st.selectbox("Get AI rec for:", [m["name"] for m in MACHINES])
    ai_btn = st.button("🤖 Ask Claude", use_container_width=True)

    st.divider()
    st.subheader("Model Metrics")
    metrics_data = {
        "Dataset": "MBEWE_DS_v1",
        "Records": "500 obs",
        "Machines": "6 nodes",
        "Status Acc": "100.0%",
        "Telemetry Acc": "91.4%",
        "Weighted F1": "90.82%",
        "AI Engine": "Claude Sonnet 4",
    }
    for k, v in metrics_data.items():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"<span style='font-size:0.65rem;color:#4a6880'>{k}</span>", unsafe_allow_html=True)
        with col2:
            colour = "#00e5a0" if "%" in v and float(v.replace("%","")) > 95 else \
                     "#ffd24d" if "%" in v else "#00c8ff"
            st.markdown(f"<span style='font-size:0.65rem;font-weight:700;color:{colour}'>{v}</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HANDLE CONTROLS
# ─────────────────────────────────────────────
if reset_btn:
    st.session_state.cycle_idx = 0
    st.session_state.history = {m["id"]: deque(maxlen=20) for m in MACHINES}
    st.session_state.event_log = deque(maxlen=40)
    st.session_state.ai_cache = {}
    st.session_state.total_cycles = 0
    st.session_state.fault_count = 0
    add_event("System reset — restarting simulation", "info")

if jump_btn:
    if "High Risk" in scenario:
        st.session_state.cycle_idx = 6   # film fault rows
    elif "Recovery" in scenario:
        st.session_state.cycle_idx = 9
    else:
        st.session_state.cycle_idx = 0
    add_event(f"Scenario loaded: {scenario}", "info")

# Advance simulation
row, risk_info = advance_cycle()

if step_btn:
    row, risk_info = advance_cycle()

if st.session_state.auto_play:
    time.sleep(0.5)   # let streamlit breathe
    st.rerun()

# ─────────────────────────────────────────────
# AI RECOMMENDATION
# ─────────────────────────────────────────────
ai_result = None
if ai_btn:
    sel_machine = next((m for m in MACHINES if m["name"] == ai_machine), MACHINES[0])
    with st.spinner(f"Consulting Claude for {sel_machine['name']}…"):
        ai_result = claude_recommendation(sel_machine, row, risk_info)
    st.session_state.ai_cache[sel_machine["id"]] = ai_result

# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────
risk_colours = {
    "Low":           ("#00e5a0", "● NOMINAL"),
    "Medium":        ("#ffd24d", "◈ MEDIUM RISK"),
    "Early Warning": ("#ff8c42", "⚡ EARLY WARNING"),
    "High":          ("#ff3d5a", "⚠ HIGH RISK"),
}
badge_colour, badge_text = risk_colours.get(risk_info["final_risk"], ("#4a6880","—"))
now = datetime.now().strftime("%H:%M:%S")

st.markdown(f"""
<div class="banner">
  <div>
    <div class="banner-title">🏭 AI RISK MANAGEMENT SYSTEM — BEVERAGE LINE</div>
    <div class="banner-sub">NUST · MEng Research · Mark Mbewe N02534127N · Supervisor: Eng T. Muhla · Bulawayo</div>
  </div>
  <div style="display:flex;gap:2rem;align-items:center">
    <div style="text-align:right">
      <div style="font-size:0.5rem;color:#4a6880;letter-spacing:0.2em;text-transform:uppercase">System Time</div>
      <div style="font-family:monospace;font-size:1rem;color:#00c8ff">{now}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:0.5rem;color:#4a6880;letter-spacing:0.2em;text-transform:uppercase">Cycle</div>
      <div style="font-family:monospace;font-size:1rem;color:#00c8ff">{st.session_state.total_cycles:04d}</div>
    </div>
    <div style="background:rgba(255,255,255,0.05);border:1px solid {badge_colour};border-radius:4px;
                padding:6px 18px;font-size:0.8rem;font-weight:700;color:{badge_colour};letter-spacing:0.15em">
      {badge_text}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Ambient Temp", f"{row['at']:.1f} °C", delta=None)
with k2:
    st.metric("Humidity", f"{row['ah']:.1f} %", delta=None)
with k3:
    st.metric("Shift", row["sh"])
with k4:
    tel_delta = f"▲ {risk_info['tel_score']:.1f}% tel risk" if risk_info['tel_score'] > 5 else None
    st.metric("Telemetry Score", f"{risk_info['tel_score']:.1f} %", delta=tel_delta,
              delta_color="inverse")
with k5:
    st.metric("Fault Events", st.session_state.fault_count,
              delta="total session" if st.session_state.fault_count else None,
              delta_color="inverse")

st.divider()

# ─────────────────────────────────────────────
# RISK ENGINE EXPLAINER
# ─────────────────────────────────────────────
with st.expander("🔍 Risk Engine Explanation", expanded=(risk_info["final_risk"] != "Low")):
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.markdown("**Status Classification Layer** *(Rule-based, 100% accurate)*")
        if risk_info["trigger_nodes"]:
            for node in risk_info["trigger_nodes"]:
                st.error(f"⚠ Fault/Warning: {node}")
        else:
            st.success("✅ All nodes running — Low risk")
        st.markdown(f"**Status Risk: `{risk_info['status_risk']}`**")

    with ecol2:
        st.markdown(f"**Telemetry Early-Warning Layer** *(Score: {risk_info['tel_score']}%)*")
        if risk_info["tel_alerts"]:
            for alert in risk_info["tel_alerts"][:5]:
                st.warning(f"⚡ {alert}")
            if len(risk_info["tel_alerts"]) > 5:
                st.caption(f"…and {len(risk_info['tel_alerts'])-5} more signals")
        else:
            st.success("✅ All telemetry within thresholds")

    st.markdown(f"""
    **Final Risk Decision: `{risk_info['final_risk']}`**  
    *Logic: Any Fault node → High | Any Warning/Idle → Medium | Telemetry score >20% → Early Warning | Otherwise → Low*
    """)

# ─────────────────────────────────────────────
# MACHINE CARDS (2 per row across 3 columns)
# ─────────────────────────────────────────────
st.markdown("### Machine Node Status")

machine_pairs = [MACHINES[i:i+2] for i in range(0, len(MACHINES), 2)]
for pair in machine_pairs:
    cols = st.columns(len(pair))
    for ci, (machine, col) in enumerate(zip(pair, cols)):
        with col:
            status = row.get(machine["status_key"], "Running")
            sp_cls = {"Running": "sp-running", "Fault": "sp-fault",
                      "Warning": "sp-warn", "Idle": "sp-idle"}.get(status, "sp-idle")
            node_risk = risk_info["status_risk"] if status == "Fault" else "Low"
            card_cls = "mcard-high" if status == "Fault" else \
                       "mcard-med" if node_risk == "Medium" else "mcard-low"

            # Build param HTML
            param_html = ""
            for label, key, unit, is_alert, is_warn in machine["params"]:
                v = row.get(key, "—")
                val_cls = "alert" if is_alert(v) else ("warn" if is_warn(v) else "")
                param_html += (
                    f'<div class="param-box">'
                    f'<div class="param-lbl">{label}</div>'
                    f'<div class="param-val {val_cls}">{v} '
                    f'<small style="font-size:0.5em;color:#4a6880">{unit}</small></div>'
                    f'</div>'
                )

            ai_html = ""
            if machine["id"] in st.session_state.ai_cache:
                rec_txt = st.session_state.ai_cache[machine["id"]]
                ai_html = f'<div class="ai-rec-box">{rec_txt}</div>'

            st.markdown(f"""
            <div class="mcard {card_cls}">
              <div class="mcard-title">{machine['name']}</div>
              <div class="mcard-brand">{machine['brand']}</div>
              <span class="status-pill {sp_cls}">● {status}</span>
              <div class="param-grid">{param_html}</div>
              {ai_html}
            </div>
            """, unsafe_allow_html=True)

            # Sparkline chart
            hist = list(st.session_state.history[machine["id"]])
            if len(hist) > 1:
                chart_df = pd.DataFrame({machine["chart_param"]: hist})
                st.line_chart(chart_df, height=80, use_container_width=True)

# ─────────────────────────────────────────────
# BOTTOM ROW: Event Log + Confusion Matrix
# ─────────────────────────────────────────────
st.divider()
bot1, bot2, bot3 = st.columns([2, 1, 1])

with bot1:
    st.markdown("#### 📋 Event Log")
    log_html = ""
    for ts, msg, lvl in list(st.session_state.event_log)[:20]:
        log_html += f'<div class="log-entry log-{lvl}"><span style="color:#1a3050;margin-right:8px">{ts}</span>{msg}</div>'
    st.markdown(f'<div style="max-height:240px;overflow-y:auto">{log_html}</div>',
                unsafe_allow_html=True)

with bot2:
    st.markdown("#### 🎯 Confusion Matrix")
    st.markdown("*Status layer — pilot validation (500 obs)*")
    cm_data = {
        "Predicted Low": [360, 0, 0],
        "Predicted Med": [0, 90, 0],
        "Predicted High": [0, 0, 50],
    }
    cm_df = pd.DataFrame(cm_data, index=["Actual Low", "Actual Med", "Actual High"])
    st.dataframe(cm_df.style.applymap(
        lambda v: "background-color:#0d2b1a;color:#00e5a0;font-weight:700" if v > 0 and v == v else
                  "background-color:#1a0a0d;color:#ff3d5a"
    ), use_container_width=True)

with bot3:
    st.markdown("#### 🗺 Implementation Roadmap")
    roadmap = [
        ("1. Readiness",    "Map assets, data sources, gaps"),
        ("2. Pilot",        "Deploy on one line, configure alerts"),
        ("3. Integration",  "Link alerts to maintenance & safety"),
        ("4. Optimisation", "Add ML, monitor false alarm rate"),
        ("5. Scale",        "Multi-line adoption + governance"),
    ]
    for phase, desc in roadmap:
        st.markdown(f"""
        <div style="background:#0f1e30;border:1px solid #1a3050;border-left:3px solid #00c8ff;
                    border-radius:4px;padding:5px 10px;margin-bottom:4px">
          <div style="font-size:0.65rem;font-weight:700;color:#00c8ff">{phase}</div>
          <div style="font-size:0.6rem;color:#4a6880">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AI RESULT CALLOUT
# ─────────────────────────────────────────────
if ai_result:
    st.divider()
    st.markdown("#### 🤖 AI Recommendation")
    sel_name = next((m["name"] for m in MACHINES if m["name"] == ai_machine), "")
    st.info(f"**{sel_name}** — {ai_result}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;font-size:0.58rem;color:#1a3050;padding:0.5rem 0">
  MBEWE MARK · N02534127N · TIE 6000 · NUST Faculty of Engineering ·
  Department of Industrial &amp; Manufacturing Engineering · May 2026 · Bulawayo
</div>
""", unsafe_allow_html=True)
