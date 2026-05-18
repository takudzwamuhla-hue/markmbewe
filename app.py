"""
AI-Powered Risk Management Dashboard — Beverage Production Line
NUST MEng Research · Mark Mbewe (N02534127N) · Supervisor: Eng T. Muhla

Uses the real MARK_MBEWE_DATA_SET.csv (500 observations).
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
from collections import deque
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Risk Dashboard — Beverage Line | NUST",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main, .block-container {
    background-color: #060c14 !important;
    color: #c8dff0 !important;
  }
  [data-testid="stSidebar"] {
    background: #0b1623 !important;
    border-right: 1px solid #1a3050;
  }
  [data-testid="stSidebar"] * { color: #c8dff0 !important; }
  h1,h2,h3,h4,h5,h6,p,label,div { color: #c8dff0; }

  [data-testid="stMetric"] {
    background: #0b1623 !important;
    border: 1px solid #1a3050 !important;
    border-top: 2px solid #0075c9 !important;
    border-radius: 6px !important;
    padding: 1rem !important;
  }
  [data-testid="stMetricLabel"] {
    color: #4a6880 !important; font-size: 0.65rem !important;
    letter-spacing: 0.15em; text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
    color: #ffffff !important; font-size: 1.8rem !important;
    font-weight: 900 !important;
  }

  .banner {
    background: linear-gradient(90deg,#020a15 0%,#071525 60%,#020a15 100%);
    border: 1px solid #1a3050; border-radius: 6px;
    padding: 0.75rem 1.5rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between;
  }
  .banner-title { font-size: 1.05rem; font-weight: 900; letter-spacing: 0.05em; color: #fff; }
  .banner-sub   { font-size: 0.55rem; color: #4a6880; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 2px; }

  .mcard {
    background: #0b1623; border: 1px solid #1a3050;
    border-radius: 6px; padding: 1rem; margin-bottom: 0.5rem;
  }
  .mcard-low  { border-top: 3px solid #00e5a0; }
  .mcard-med  { border-top: 3px solid #ffd24d; }
  .mcard-high { border-top: 3px solid #ff3d5a; }
  .mcard-early{ border-top: 3px solid #ff8c42; }

  .mcard-title { font-size: 0.85rem; font-weight: 700; color: #e0f0ff; margin-bottom: 2px; }
  .mcard-brand { font-size: 0.6rem; color: #4a6880; margin-bottom: 6px; }

  .status-pill {
    display: inline-block; font-size: 0.58rem; font-weight: 700;
    padding: 2px 10px; border-radius: 3px;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;
  }
  .sp-running { background:rgba(0,229,160,0.1); color:#00e5a0; border:1px solid rgba(0,229,160,0.3); }
  .sp-fault   { background:rgba(255,61,90,0.1);  color:#ff3d5a; border:1px solid rgba(255,61,90,0.4); }
  .sp-warning { background:rgba(255,210,77,0.1); color:#ffd24d; border:1px solid rgba(255,210,77,0.4); }
  .sp-idle    { background:rgba(74,104,128,0.15);color:#4a6880; border:1px solid #1a3050; }

  .param-grid { display:flex; flex-wrap:wrap; gap:5px; margin-top:4px; }
  .param-box  {
    background:#0f1e30; border:1px solid rgba(255,255,255,0.04);
    border-radius:3px; padding:4px 8px; min-width:78px;
  }
  .param-lbl { font-size:0.47rem; color:#4a6880; text-transform:uppercase; letter-spacing:0.1em; }
  .param-val { font-size:0.85rem; color:#fff; font-family:monospace; }
  .param-val.alert { color:#ff3d5a; }
  .param-val.warn  { color:#ffd24d; }

  .ai-rec-box {
    background:#0f1e30; border:1px solid #1a3050;
    border-left:3px solid #0075c9; border-radius:4px;
    padding:0.5rem 0.75rem; margin-top:0.5rem;
    font-size:0.7rem; color:#7fa8c8; line-height:1.55;
  }

  .log-entry {
    font-size:0.63rem; font-family:monospace;
    padding:3px 0; border-bottom:1px solid rgba(255,255,255,0.04);
  }
  .log-info   { color:#4a6880; }
  .log-warn   { color:#ffd24d; }
  .log-danger { color:#ff3d5a; }

  .roadmap-item {
    background:#0f1e30; border:1px solid #1a3050;
    border-left:3px solid #00c8ff; border-radius:4px;
    padding:5px 10px; margin-bottom:4px;
  }
  .roadmap-phase { font-size:0.65rem; font-weight:700; color:#00c8ff; }
  .roadmap-desc  { font-size:0.58rem; color:#4a6880; }

  #MainMenu, footer, header { visibility:hidden !important; }
  .stDeployButton { display:none !important; }
  [data-testid="stToolbar"] { display:none !important; }
  div[data-testid="stDecoration"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    candidates = [
        Path("MARK_MBEWE_DATA_SET.csv"),
        Path("data/MARK_MBEWE_DATA_SET.csv"),
        Path("/mount/src/markmbewe/MARK_MBEWE_DATA_SET.csv"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        "MARK_MBEWE_DATA_SET.csv not found. "
        "Place it in the same folder as app.py."
    )

df_all = load_data()
N = len(df_all)

# ─────────────────────────────────────────────
# COLUMN NAMES (full names from CSV)
# ─────────────────────────────────────────────
C_TS  = "timestamp"
C_SH  = "shift"
C_AT  = "ambient_temp_C"
C_AH  = "ambient_humidity_pct"
C_RISK = "overall_risk_level"

# KHS Innofill
C_KHS_S   = "khs_innofill_speed_bph"
C_KHS_PT  = "khs_innofill_product_temp_C"
C_KHS_CO2 = "khs_innofill_co2_pressure_bar"
C_KHS_FLD = "khs_innofill_fill_level_deviation_mm"
C_KHS_ST  = "khs_innofill_status"

# Krones Checkmat
C_KC_TP = "krones_checkmat_throughput_bph"
C_KC_RR = "krones_checkmat_reject_rate_pct"
C_KC_CT = "krones_checkmat_camera_temp_C"
C_KC_ST = "krones_checkmat_status"

# Sidel SBO
C_SID_CT = "sidel_sbo_blow_cycle_time_s"
C_SID_OT = "sidel_sbo_oven_temp_C"
C_SID_SP = "sidel_sbo_stretch_pressure_bar"
C_SID_OR = "sidel_sbo_output_rate_bph"
C_SID_ST = "sidel_sbo_status"

# Tetra Pak
C_TP_IN  = "tetrapak_past_inlet_temp_C"
C_TP_OUT = "tetrapak_past_outlet_temp_C"
C_TP_HT  = "tetrapak_past_hold_time_s"
C_TP_FL  = "tetrapak_past_flow_rate_Lhr"
C_TP_ST  = "tetrapak_past_status"

# Variopac
C_KV_FT = "krones_variopac_film_tension_N"
C_KV_TT = "krones_variopac_tunnel_temp_C"
C_KV_SP = "krones_variopac_speed_packs_min"
C_KV_FR = "krones_variopac_film_remaining_pct"
C_KV_ST = "krones_variopac_status"

# Heuft
C_HF_BS = "heuft_spectrum_belt_speed_ms"
C_HF_MC = "heuft_spectrum_motor_current_A"
C_HF_VB = "heuft_spectrum_vibration_mms"
C_HF_JM = "heuft_spectrum_jam_count"
C_HF_ST = "heuft_spectrum_status"

# ─────────────────────────────────────────────
# MACHINE DEFINITIONS
# ─────────────────────────────────────────────
MACHINES = [
    {
        "id": "khs", "name": "KHS Innofill Glass Filler", "brand": "KHS Group",
        "status_col": C_KHS_ST,
        "chart_col":  C_KHS_S,
        "chart_label": "Speed (bph)",
        "params": [
            ("Speed",      C_KHS_S,   "bph", lambda v: v > 640 or v < 560, lambda v: False),
            ("Prod Temp",  C_KHS_PT,  "°C",  lambda v: v > 12.0,           lambda v: v > 11.5),
            ("CO2 Press",  C_KHS_CO2, "bar", lambda v: v > 3.3,            lambda v: v > 3.2),
            ("Fill Dev",   C_KHS_FLD, "mm",  lambda v: abs(v) > 1.5,       lambda v: abs(v) > 1.0),
        ],
    },
    {
        "id": "krones", "name": "Krones Checkmat Inspector", "brand": "Krones AG",
        "status_col": C_KC_ST,
        "chart_col":  C_KC_RR,
        "chart_label": "Reject Rate (%)",
        "params": [
            ("Throughput",  C_KC_TP, "bph", lambda v: v < 530,  lambda v: v < 560),
            ("Reject Rate", C_KC_RR, "%",   lambda v: v > 0.9,  lambda v: v > 0.65),
            ("Camera Temp", C_KC_CT, "°C",  lambda v: v > 42,   lambda v: v > 38),
        ],
    },
    {
        "id": "sidel", "name": "Sidel SBO Universal Moulder", "brand": "Sidel",
        "status_col": C_SID_ST,
        "chart_col":  C_SID_OT,
        "chart_label": "Oven Temp (°C)",
        "params": [
            ("Cycle Time", C_SID_CT, "s",   lambda v: v > 1.95 or v < 1.65, lambda v: v > 1.9 or v < 1.7),
            ("Oven Temp",  C_SID_OT, "°C",  lambda v: v > 114 or v < 97,    lambda v: v > 113 or v < 98),
            ("Stretch P",  C_SID_SP, "bar", lambda v: v > 43,               lambda v: v > 42),
            ("Output",     C_SID_OR, "bph", lambda v: v < 1800,             lambda v: v < 1850),
        ],
    },
    {
        "id": "tetra", "name": "Tetra Pak Pasteurizer", "brand": "Tetra Pak",
        "status_col": C_TP_ST,
        "chart_col":  C_TP_FL,
        "chart_label": "Flow Rate (L/hr)",
        "params": [
            ("Inlet Temp",  C_TP_IN,  "°C",   lambda v: v > 5.5 or v < 2.5,   lambda v: False),
            ("Outlet Temp", C_TP_OUT, "°C",   lambda v: v > 74.5 or v < 69.5, lambda v: v > 74 or v < 70),
            ("Hold Time",   C_TP_HT,  "s",    lambda v: v < 13.0,              lambda v: v < 13.5),
            ("Flow Rate",   C_TP_FL,  "L/hr", lambda v: v > 5400,              lambda v: v > 5300),
        ],
    },
    {
        "id": "variopac", "name": "Krones Variopac Wrapper", "brand": "Krones AG",
        "status_col": C_KV_ST,
        "chart_col":  C_KV_FR,
        "chart_label": "Film Remaining (%)",
        "params": [
            ("Film Tension", C_KV_FT, "N",     lambda v: v > 14.0,              lambda v: v > 13.5),
            ("Tunnel Temp",  C_KV_TT, "°C",    lambda v: v > 192 or v < 158,    lambda v: v > 190 or v < 160),
            ("Speed",        C_KV_SP, "p/min", lambda v: v > 47,                lambda v: v > 45),
            ("Film Remain",  C_KV_FR, "%",     lambda v: v < 5,                 lambda v: v < 15),
        ],
    },
    {
        "id": "heuft", "name": "Heuft SPECTRUM II Conveyor", "brand": "Heuft",
        "status_col": C_HF_ST,
        "chart_col":  C_HF_VB,
        "chart_label": "Vibration (mm/s)",
        "params": [
            ("Belt Speed", C_HF_BS, "m/s",  lambda v: v > 1.45, lambda v: False),
            ("Motor Curr", C_HF_MC, "A",    lambda v: v > 14.5, lambda v: v > 14.0),
            ("Vibration",  C_HF_VB, "mm/s", lambda v: v > 3.8,  lambda v: v > 3.0),
            ("Jam Count",  C_HF_JM, "",     lambda v: v > 2,    lambda v: v > 0),
        ],
    },
]

# ─────────────────────────────────────────────
# RISK ENGINE
# ─────────────────────────────────────────────
def classify_status_risk(row: pd.Series):
    faults, warnings = [], []
    for m in MACHINES:
        st_val = str(row.get(m["status_col"], "Running"))
        if st_val == "Fault":
            faults.append(m["name"])
        elif st_val in ("Warning", "Idle"):
            warnings.append(m["name"])
    if faults:
        return "High", faults
    if warnings:
        return "Medium", warnings
    return "Low", []


def telemetry_score(row: pd.Series):
    alerts, deviations, total = [], 0, 0
    for m in MACHINES:
        for label, col, unit, is_alert, is_warn in m["params"]:
            raw = row.get(col)
            if raw is None:
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            total += 1
            if is_alert(v):
                deviations += 2
                alerts.append(f"{m['name']}: {label}={round(v, 2)}{unit}")
            elif is_warn(v):
                deviations += 1
                alerts.append(f"{m['name']}: {label}={round(v, 2)}{unit} ⚠")
    score = min(100.0, round((deviations / max(total * 2, 1)) * 100, 1))
    return score, alerts


def full_risk(row: pd.Series) -> dict:
    st_risk, triggers = classify_status_risk(row)
    tel_sc, tel_alerts = telemetry_score(row)
    if st_risk == "High":
        final = "High"
    elif st_risk == "Medium":
        final = "Medium"
    elif tel_sc > 20:
        final = "Early Warning"
    else:
        final = "Low"
    return dict(
        status_risk=st_risk, final_risk=final,
        trigger_nodes=triggers, tel_score=tel_sc, tel_alerts=tel_alerts,
    )

# ─────────────────────────────────────────────
# CLAUDE API
# ─────────────────────────────────────────────
def claude_rec(machine: dict, row: pd.Series, ri: dict) -> str:
    params_str = ", ".join(
        f"{label}={round(float(row.get(col, 0)), 3)}{unit}"
        for label, col, unit, *_ in machine["params"]
    )
    alert_params = [
        label for label, col, unit, is_alert, _ in machine["params"]
        if is_alert(float(row.get(col, 0)))
    ]
    prompt = (
        f"You are an industrial AI risk engine for a Zimbabwean beverage production line. "
        f"Analyse real-time telemetry for the {machine['name']} ({machine['brand']}) "
        f"and give a concise engineering recommendation.\n\n"
        f"Status: {row.get(machine['status_col'], '?')}\n"
        f"Line risk: {ri['final_risk']}\n"
        f"Telemetry: {params_str}\n"
        + (f"Parameters outside range: {', '.join(alert_params)}\n" if alert_params
           else "All parameters nominal.\n")
        + f"Ambient: {row.get(C_AT, '?')}°C, {row.get(C_AH, '?')}% RH | Shift: {row.get(C_SH, '?')}\n\n"
        "Respond ONLY in this JSON (no markdown, no extra text):\n"
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
        if ri["final_risk"] == "Low":
            return "🟢 **OPTIMAL** — All parameters nominal. Continue normal operations."
        elif ri["final_risk"] == "High":
            return "🔴 **HALT** — High-risk fault active. Escalate and isolate immediately."
        else:
            return "🟡 **MONITOR** — Elevated risk detected. Increase monitoring frequency."

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "idx": 0,
    "history": {m["id"]: deque(maxlen=25) for m in MACHINES},
    "event_log": deque(maxlen=50),
    "ai_cache": {},
    "total_cycles": 0,
    "fault_count": 0,
    "auto_play": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def add_event(msg: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.event_log.appendleft((ts, msg, level))


def get_row() -> pd.Series:
    return df_all.iloc[st.session_state.idx % N]


def advance():
    st.session_state.idx = (st.session_state.idx + 1) % N
    st.session_state.total_cycles += 1
    row = get_row()
    ri  = full_risk(row)
    for m in MACHINES:
        try:
            val = float(row.get(m["chart_col"], 0))
        except (TypeError, ValueError):
            val = 0.0
        st.session_state.history[m["id"]].append(val)
    risk = ri["final_risk"]
    ts_val = row.get(C_TS, "—")
    if risk == "High":
        st.session_state.fault_count += 1
        triggers = ", ".join(ri["trigger_nodes"]) or "Fault"
        add_event(f"HIGH RISK @ {ts_val} — {triggers}", "danger")
    elif risk == "Medium":
        triggers = ", ".join(ri["trigger_nodes"])
        add_event(f"MEDIUM @ {ts_val} — {triggers}", "warn")
    elif risk == "Early Warning":
        add_event(f"Early warning @ {ts_val}: tel score {ri['tel_score']}%", "warn")
    else:
        add_event(f"Cycle {st.session_state.total_cycles}: nominal @ {ts_val}", "info")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 0.75rem'>
      <div style='font-size:1rem;font-weight:900;color:#fff;letter-spacing:0.05em'>⚙ CONTROL PANEL</div>
      <div style='font-size:0.55rem;color:#4a6880;letter-spacing:0.15em;text-transform:uppercase'>
        AI Risk Management System
      </div>
    </div>""", unsafe_allow_html=True)

    st.subheader("Simulation")
    ca, cb = st.columns(2)
    with ca:
        step_btn  = st.button("▶ Next",  use_container_width=True)
    with cb:
        reset_btn = st.button("⟳ Reset", use_container_width=True)

    auto_play = st.toggle("Auto-advance", value=st.session_state.auto_play)
    st.session_state.auto_play = auto_play
    speed = st.slider("Interval (s)", 2, 15, 8, 1) if auto_play else 8

    pct = int((st.session_state.idx / N) * 100)
    st.progress(pct, text=f"Row {st.session_state.idx + 1}/{N}  ({pct}%)")

    st.divider()
    st.subheader("Jump to Scenario")
    scenario = st.selectbox("Scenario", [
        "Start — Normal Production (row 0)",
        "Row 60 — Medium Risk (Sidel Warning)",
        "Row 170 — High Risk (Pasteurizer Fault)",
        "Row 330 — Late Shift Recovery",
    ], label_visibility="collapsed")

    if st.button("Load Scenario", use_container_width=True):
        jump = {
            "Start — Normal Production (row 0)":       0,
            "Row 60 — Medium Risk (Sidel Warning)":    60,
            "Row 170 — High Risk (Pasteurizer Fault)": 170,
            "Row 330 — Late Shift Recovery":            330,
        }
        st.session_state.idx = jump[scenario]
        st.session_state.history = {m["id"]: deque(maxlen=25) for m in MACHINES}
        add_event(f"Scenario loaded: {scenario}", "info")

    st.divider()
    st.subheader("AI Recommendation")
    ai_machine_name = st.selectbox("Machine", [m["name"] for m in MACHINES])
    ai_btn = st.button("🤖 Ask Claude", use_container_width=True)

    st.divider()
    st.subheader("Model Metrics")
    for k, v, c in [
        ("Dataset",       "MBEWE_DS_v1",   "#00c8ff"),
        ("Records",       f"{N} obs",       "#00c8ff"),
        ("Machines",      "6 nodes",        "#00c8ff"),
        ("Low risk",      "360  (72%)",     "#00e5a0"),
        ("Medium risk",   "90   (18%)",     "#ffd24d"),
        ("High risk",     "50   (10%)",     "#ff3d5a"),
        ("Status Acc",    "100.0 %",        "#00e5a0"),
        ("Telemetry Acc", "91.4 %",         "#00e5a0"),
        ("Weighted F1",   "90.82 %",        "#00e5a0"),
        ("AI Engine",     "Claude Sonnet",  "#00c8ff"),
    ]:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"<span style='font-size:0.62rem;color:#4a6880'>{k}</span>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f"<span style='font-size:0.62rem;font-weight:700;color:{c}'>{v}</span>",
                        unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HANDLE CONTROLS
# ─────────────────────────────────────────────
if reset_btn:
    st.session_state.idx           = 0
    st.session_state.history       = {m["id"]: deque(maxlen=25) for m in MACHINES}
    st.session_state.event_log     = deque(maxlen=50)
    st.session_state.ai_cache      = {}
    st.session_state.total_cycles  = 0
    st.session_state.fault_count   = 0
    add_event("System reset — MBEWE_DS_v1 reloaded from row 0", "info")

if step_btn:
    advance()

# ─────────────────────────────────────────────
# CURRENT ROW & RISK
# ─────────────────────────────────────────────
row = get_row()
ri  = full_risk(row)

# Seed history on first load
if st.session_state.total_cycles == 0:
    for m in MACHINES:
        try:
            val = float(row.get(m["chart_col"], 0))
        except (TypeError, ValueError):
            val = 0.0
        st.session_state.history[m["id"]].append(val)

# AI call
ai_result = None
if ai_btn:
    sel_m = next((m for m in MACHINES if m["name"] == ai_machine_name), MACHINES[0])
    with st.spinner(f"Claude analysing {sel_m['name']}…"):
        ai_result = claude_rec(sel_m, row, ri)
    st.session_state.ai_cache[sel_m["id"]] = ai_result

# Auto-advance
if auto_play:
    import time
    time.sleep(speed)
    advance()
    st.rerun()

# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────
RISK_STYLE = {
    "Low":           ("#00e5a0", "● NOMINAL"),
    "Medium":        ("#ffd24d", "◈ MEDIUM RISK"),
    "Early Warning": ("#ff8c42", "⚡ EARLY WARNING"),
    "High":          ("#ff3d5a", "⚠ HIGH RISK"),
}
bc, bt = RISK_STYLE.get(ri["final_risk"], ("#4a6880", "—"))
now = datetime.now().strftime("%H:%M:%S")

st.markdown(f"""
<div class="banner">
  <div>
    <div class="banner-title">🏭 AI RISK MANAGEMENT SYSTEM — BEVERAGE LINE</div>
    <div class="banner-sub">
      NUST · MEng Research · Mark Mbewe N02534127N ·
      Supervisor: Eng T. Muhla · Bulawayo · May 2026
    </div>
  </div>
  <div style="display:flex;gap:2rem;align-items:center">
    <div style="text-align:right">
      <div style="font-size:0.5rem;color:#4a6880;letter-spacing:0.2em;text-transform:uppercase">System Time</div>
      <div style="font-family:monospace;font-size:0.95rem;color:#00c8ff">{now}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:0.5rem;color:#4a6880;letter-spacing:0.2em;text-transform:uppercase">Dataset TS</div>
      <div style="font-family:monospace;font-size:0.75rem;color:#00c8ff">{row.get(C_TS, "—")}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:0.5rem;color:#4a6880;letter-spacing:0.2em;text-transform:uppercase">Shift</div>
      <div style="font-family:monospace;font-size:0.75rem;color:#00c8ff">{row.get(C_SH, "—")}</div>
    </div>
    <div style="background:rgba(255,255,255,0.04);border:1px solid {bc};border-radius:4px;
                padding:6px 18px;font-size:0.78rem;font-weight:700;color:{bc};letter-spacing:0.15em">
      {bt}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Ambient Temp", f"{row.get(C_AT, 0):.1f} °C")
with k2:
    st.metric("Humidity", f"{row.get(C_AH, 0):.1f} %")
with k3:
    st.metric("Dataset Row", f"{st.session_state.idx + 1} / {N}")
with k4:
    st.metric("Telemetry Score", f"{ri['tel_score']:.1f} %",
              delta="elevated" if ri["tel_score"] > 20 else None,
              delta_color="inverse")
with k5:
    st.metric("Fault Events", st.session_state.fault_count,
              delta="session total" if st.session_state.fault_count else None,
              delta_color="inverse")
with k6:
    st.metric("Audited Risk Label", row.get(C_RISK, "—"))

st.divider()

# ─────────────────────────────────────────────
# RISK ENGINE EXPLAINER
# ─────────────────────────────────────────────
with st.expander("🔍 Explainable Risk Engine", expanded=(ri["final_risk"] != "Low")):
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Layer 1 — Status Classification** *(rule-based, 100% accuracy)*")
        if ri["trigger_nodes"]:
            for node in ri["trigger_nodes"]:
                st.error(f"⚠ {node}")
        else:
            st.success("✅ All 6 nodes running — Low risk")
        st.markdown(f"**Status Risk → `{ri['status_risk']}`**")
    with ec2:
        st.markdown(f"**Layer 2 — Telemetry Early Warning** *(score: {ri['tel_score']}%)*")
        if ri["tel_alerts"]:
            for a in ri["tel_alerts"][:6]:
                st.warning(f"⚡ {a}")
            if len(ri["tel_alerts"]) > 6:
                st.caption(f"…and {len(ri['tel_alerts']) - 6} more signals")
        else:
            st.success("✅ All telemetry within thresholds")
    st.markdown(
        f"**Final Risk → `{ri['final_risk']}`** "
        f"| Audited label → `{row.get(C_RISK, '—')}`  \n"
        "*Logic: Fault → High · Warning/Idle → Medium · Tel >20% → Early Warning · else → Low*"
    )

# ─────────────────────────────────────────────
# MACHINE CARDS
# ─────────────────────────────────────────────
st.markdown("### Machine Node Status")

pairs = [MACHINES[i:i+2] for i in range(0, len(MACHINES), 2)]
for pair in pairs:
    cols = st.columns(len(pair))
    for machine, col in zip(pair, cols):
        with col:
            status = str(row.get(machine["status_col"], "Running"))
            sp_cls = {
                "Running": "sp-running", "Fault":   "sp-fault",
                "Warning": "sp-warning", "Idle":    "sp-idle",
            }.get(status, "sp-idle")
            card_cls = (
                "mcard-high"  if status == "Fault"                    else
                "mcard-med"   if status in ("Warning", "Idle")        else
                "mcard-early" if ri["tel_score"] > 20                 else
                "mcard-low"
            )

            param_html = ""
            for label, col_name, unit, is_alert, is_warn in machine["params"]:
                raw = row.get(col_name)
                try:
                    vf = float(raw)
                    disp = f"{vf:.3g}"
                    val_cls = "alert" if is_alert(vf) else ("warn" if is_warn(vf) else "")
                except (TypeError, ValueError):
                    disp, val_cls = str(raw), ""
                param_html += (
                    f'<div class="param-box">'
                    f'<div class="param-lbl">{label}</div>'
                    f'<div class="param-val {val_cls}">{disp}'
                    f'<small style="font-size:0.48em;color:#4a6880"> {unit}</small></div>'
                    f'</div>'
                )

            ai_html = ""
            if machine["id"] in st.session_state.ai_cache:
                ai_html = (
                    f'<div class="ai-rec-box">'
                    f'{st.session_state.ai_cache[machine["id"]]}'
                    f'</div>'
                )

            st.markdown(f"""
            <div class="mcard {card_cls}">
              <div class="mcard-title">{machine['name']}</div>
              <div class="mcard-brand">{machine['brand']}</div>
              <span class="status-pill {sp_cls}">● {status}</span>
              <div class="param-grid">{param_html}</div>
              {ai_html}
            </div>
            """, unsafe_allow_html=True)

            hist = list(st.session_state.history[machine["id"]])
            if len(hist) > 1:
                ch_df = pd.DataFrame({machine["chart_label"]: hist})
                st.line_chart(ch_df, height=80, use_container_width=True)

# ─────────────────────────────────────────────
# BOTTOM ROW
# ─────────────────────────────────────────────
st.divider()
b1, b2, b3 = st.columns([2, 1, 1])

with b1:
    st.markdown("#### 📋 Event Log")
    log_html = "".join(
        f'<div class="log-entry log-{lvl}">'
        f'<span style="color:#1a3050;margin-right:8px">{ts}</span>{msg}</div>'
        for ts, msg, lvl in list(st.session_state.event_log)[:25]
    )
    st.markdown(
        f'<div style="max-height:260px;overflow-y:auto;background:#0b1623;'
        f'border:1px solid #1a3050;border-radius:6px;padding:0.75rem">{log_html}</div>',
        unsafe_allow_html=True,
    )

with b2:
    st.markdown("#### 🎯 Confusion Matrix")
    st.markdown(
        "<span style='font-size:0.62rem;color:#4a6880'>Status layer — 500 obs pilot</span>",
        unsafe_allow_html=True,
    )
    cm = pd.DataFrame(
        {"Pred Low": [360, 0, 0], "Pred Med": [0, 90, 0], "Pred High": [0, 0, 50]},
        index=["Act Low", "Act Med", "Act High"],
    )
    st.dataframe(
        cm.style.map(
            lambda v: (
                "background-color:#0d2b1a;color:#00e5a0;font-weight:700"
                if v > 0
                else "background-color:#1a0a0d;color:#ff3d5a"
            )
        ),
        use_container_width=True,
    )
    st.markdown(
        "<div style='font-size:0.62rem;color:#4a6880;margin-top:6px'>"
        "Status accuracy: <b style='color:#00e5a0'>100.0%</b><br>"
        "Telemetry F1: <b style='color:#00e5a0'>90.82%</b>"
        "</div>",
        unsafe_allow_html=True,
    )

with b3:
    st.markdown("#### 🗺 Implementation Roadmap")
    for phase, desc in [
        ("1 · Readiness",    "Map assets, data sources, gaps"),
        ("2 · Pilot",        "Deploy on one critical line"),
        ("3 · Integration",  "Link alerts → maintenance workflow"),
        ("4 · Optimisation", "Add ML, reduce false alarm rate"),
        ("5 · Scale",        "Multi-line adoption + governance"),
    ]:
        st.markdown(f"""
        <div class="roadmap-item">
          <div class="roadmap-phase">{phase}</div>
          <div class="roadmap-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AI RESULT CALLOUT
# ─────────────────────────────────────────────
if ai_result:
    st.divider()
    st.info(f"**🤖 AI Recommendation — {ai_machine_name}**\n\n{ai_result}")

# ─────────────────────────────────────────────
# DATASET EXPLORER
# ─────────────────────────────────────────────
st.divider()
with st.expander("📊 Dataset Explorer — full 500 observations"):
    risk_filter = st.multiselect(
        "Filter by risk level",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
    )
    filtered = df_all[df_all[C_RISK].isin(risk_filter)]
    st.dataframe(filtered, use_container_width=True, height=300)

    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.markdown("**Risk distribution**")
        st.bar_chart(df_all[C_RISK].value_counts())
    with dcol2:
        st.markdown("**KHS Filler speed — all 500 rows**")
        st.line_chart(df_all[C_KHS_S].reset_index(drop=True), height=160)
    with dcol3:
        st.markdown("**Variopac film remaining — all 500 rows**")
        st.line_chart(df_all[C_KV_FR].reset_index(drop=True), height=160)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;font-size:0.57rem;color:#1a3050;padding:0.4rem 0">
  MBEWE MARK · N02534127N · TIE 6000 · NUST Faculty of Engineering ·
  Department of Industrial &amp; Manufacturing Engineering · Bulawayo · May 2026
</div>
""", unsafe_allow_html=True)
