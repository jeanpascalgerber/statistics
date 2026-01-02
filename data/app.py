import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import time
import urllib3

# Warnungen unterdrücken
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------------------------
# 1. DESIGN: MISSION CONTROL (FINAL)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AURORA MISSION CONTROL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    /* DARK THEME */
    .stApp { 
        background-color: #050505; 
        color: #00ff41; 
        font-family: 'JetBrains Mono', monospace; 
    }
    
    footer, header, .stDeployButton { visibility: hidden; }
    
    /* METRIC BOXES */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #333;
        border-left: 3px solid #333;
        padding: 15px;
    }
    div[data-testid="stMetric"]:hover {
        border-left: 3px solid #00ff41;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.1);
    }
    
    /* LABELS */
    div[data-testid="stMetricLabel"] p {
        color: #666 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* VALUES */
    div[data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
        font-size: 26px !important;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* HEADER STYLE */
    .hud-header {
        border-bottom: 1px solid #333;
        margin-bottom: 30px;
        padding-bottom: 10px;
        color: #666;
        font-size: 12px;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    
    /* SECTION TITLES */
    .section-title {
        color: #00ff41;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    /* TABS STYLE */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0a0a0a;
        border: 1px solid #333;
        color: #666;
        font-size: 12px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff41 !important;
        color: black !important;
        font-weight: bold;
    }

    .stSlider > div > div > div > div { background-color: #333; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA ENGINE
# -----------------------------------------------------------------------------
def get_live_telemetry():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r_mag = requests.get("https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json", headers=headers, timeout=5, verify=False).json()
        r_plasma = requests.get("https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json", headers=headers, timeout=5, verify=False).json()
        r_kp = requests.get("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json", headers=headers, timeout=5, verify=False).json()

        mag = r_mag[-1]
        plasma = r_plasma[-1]
        kp = r_kp[-1]

        return {
            'bz': float(mag[3]),
            'by': float(mag[2]),
            'bt': float(mag[4]),
            'speed': float(plasma[2]),
            'density': float(plasma[1]),
            'temp': float(plasma[3]),
            'kp': float(kp[1]),
            'time': mag[0],
            'status': 'ONLINE',
            'msg': 'OK'
        }
    except Exception as e:
        return {'status': 'OFFLINE', 'kp': 0.0, 'msg': str(e)}

# -----------------------------------------------------------------------------
# 3. VISUALS
# -----------------------------------------------------------------------------
def create_gauge(prob):
    kp_val = prob * 9.0
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = kp_val,
        number = {'suffix': " KP", 'font': {'color': "white", 'family': "JetBrains Mono"}},
        gauge = {
            'axis': {'range': [0, 9], 'tickcolor': "#444"},
            'bar': {'color': "#00ff41"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 4], 'color': 'rgba(50, 50, 50, 0.5)'}, 
                {'range': [4, 6], 'color': 'rgba(100, 100, 0, 0.5)'}, 
                {'range': [6, 9], 'color': 'rgba(100, 0, 0, 0.5)'}
            ],
            'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.8, 'value': kp_val}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=20), height=180)
    return fig

# -----------------------------------------------------------------------------
# 4. MODEL ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def init_model():
    try:
        cols = ['Year', 'DOY', 'Hour', 'By', 'Bz', 'Density', 'Speed', 'Kp_10']
        df = pd.read_csv('omni_data.txt', delim_whitespace=True, names=cols)
        df = df.replace([999.9, 9999.99, 999.99, 9999, 99.9], np.nan)
        df['Kp'] = df['Kp_10'] / 10.0
        
        feats = ['Bz', 'By', 'Speed', 'Density']
        df_m = df[feats + ['Kp']].dropna().copy()
        
        df_m['Target'] = (df_m['Kp'] >= 5).astype(int)
        df_m['Target_Future'] = df_m['Target'].shift(-3)
        df_m = df_m.dropna()
        
        X_tr, X_te, y_tr, y_te = train_test_split(df_m[feats], df_m['Target_Future'], test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
        model.fit(X_tr, y_tr)
        
        acc = accuracy_score(y_te, model.predict(X_te))
        return model, df, acc, feats
    except:
        return None, None, 0, []

model, df_hist, score, f_names = init_model()
if not model:
    st.error("DATABASE ERROR: omni_data.txt NOT FOUND")
    st.stop()

# # -----------------------------------------------------------------------------
# 5. INITIALIZATION
# -----------------------------------------------------------------------------
telemetry = get_live_telemetry()

# Sanity Check
# Wir prüfen, ob die Sensoren "Quatsch" (9999) senden, bevor das Modell es sieht.
if telemetry['status'] == 'ONLINE':
    # 1. Speed Check (Werte über 2000 km/s sind physikalisch fast unmöglich)
    if telemetry['speed'] > 2000 or telemetry['speed'] < 0:
        telemetry['speed'] = 400.0 # Reset auf ruhigen Standardwert
        st.toast("⚠️ Sensor-Fehler korrigiert: Speed > 2000 km/s", icon="🔧")
    
    # 2. Density Check (Werte über 100 sind meist Fehler)
    if telemetry['density'] > 100 or telemetry['density'] < 0:
        telemetry['density'] = 5.0
        st.toast("⚠️ Sensor-Fehler korrigiert: Density unplausibel", icon="🔧")
        
    # 3. Bz Check (Werte über 100 nT sind Fehler)
    if abs(telemetry['bz']) > 100:
        telemetry['bz'] = 0.0
        st.toast("⚠️ Sensor-Fehler korrigiert: Bz > 100 nT", icon="🔧")
# --- SANITY CHECK ENDE ---

is_online = telemetry['status'] == 'ONLINE'

if is_online:
    current = telemetry
else:
    last = df_hist.iloc[-1]
    current = {'bz': float(last['Bz']), 'by': float(last['By']), 'bt': 5.0, 'speed': float(last['Speed']), 'density': float(last['Density']), 'temp': 100000.0, 'kp': 0.0}

for k in ['bz', 'by', 'speed', 'density', 'temp']:
    if k not in st.session_state: st.session_state[k] = current.get(k, 0.0)

# -----------------------------------------------------------------------------
# 6. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.markdown("### /// SYSTEM CONTROL")

if st.sidebar.button("RELOAD TELEMETRY"):
    st.cache_data.clear()
    st.rerun()

if not is_online:
    st.sidebar.error("OFFLINE MODE")
else:
    try: clean_time = str(telemetry['time']).split(' ')[1][:5]
    except: clean_time = "SYNC"
    st.sidebar.success(f"LINK ESTABLISHED: {clean_time} UTC")

st.sidebar.markdown("---")
st.sidebar.markdown("MANUAL OVERRIDE")

i_sp = st.sidebar.slider("VELOCITY (km/s)", 200.0, 1500.0, key='speed')
i_dn = st.sidebar.slider("DENSITY (p/cm3)", 0.1, 100.0, key='density')
i_tp = st.sidebar.slider("TEMP (K)", 1000.0, 1000000.0, key='temp')
i_bz = st.sidebar.slider("IMF BZ (nT)", -50.0, 50.0, key='bz')
i_by = st.sidebar.slider("IMF BY (nT)", -50.0, 50.0, key='by')

prob = model.predict_proba(pd.DataFrame([[i_bz, i_by, i_sp, i_dn]], columns=f_names))[0, 1]
dyn_pressure = 1.6726e-6 * i_dn * (i_sp**2)

# -----------------------------------------------------------------------------
# 7. DASHBOARD
# -----------------------------------------------------------------------------
st.markdown("<div class='hud-header'>/// MISSION CONTROL CENTER // AURORA ANALYTICS</div>", unsafe_allow_html=True)

# ROW 1: PLASMA
st.markdown("<div class='section-title'>PLASMA DYNAMICS</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("SOLAR WIND SPEED", f"{i_sp:.0f}", "km/s")
c2.metric("ION DENSITY", f"{i_dn:.1f}", "p/cm³")
c3.metric("DYNAMIC PRESSURE", f"{dyn_pressure:.2f}", "nPa")
c4.metric("PLASMA TEMP", f"{i_tp/1000:.0f}k", "K")

st.markdown("<br>", unsafe_allow_html=True)

# ROW 2: MAGNETIC FIELD
st.markdown("<div class='section-title'>INTERPLANETARY MAGNETIC FIELD (IMF)</div>", unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)
c5.metric("BZ (NORTH/SOUTH)", f"{i_bz:.1f}", "nT", delta_color="off")
c6.metric("BY (EAST/WEST)", f"{i_by:.1f}", "nT", delta_color="off")
c7.metric("BT (TOTAL)", f"{current.get('bt', 0):.1f}", "nT", delta_color="off")

status = "QUIET"
if prob > 0.8: status = "CRITICAL STORM"
elif prob > 0.5: status = "WARNING"
c8.metric("ALERT STATUS", status)

st.markdown("---")

# ROW 3: VISUALS (MIT TABS FÜR SONNE/ERDE)
col_L, col_R = st.columns([1, 2])

with col_L:
    st.markdown("<div class='section-title'>PREDICTIVE MODEL (RF)</div>", unsafe_allow_html=True)
    st.plotly_chart(create_gauge(prob), use_container_width=True, key="gauge")
    
    real_kp = current.get('kp', 0.0)
    st.metric("NOAA GROUND TRUTH", f"{real_kp:.2f}", "KP")
    st.caption(f"MODEL ACCURACY: {score:.1%} | ALGORITHM: Random Forest")

with col_R:
    # HIER IST DAS NEUE FEATURE: TABS
    tab1, tab2 = st.tabs(["TARGET: EARTH (IMPACT)", "SOURCE: SUN (ORIGIN)"])
    
    # 1. Bild: Erd-Simulation (NOAA)
    with tab1:
        ts = int(time.time())
        st.image(f"https://services.swpc.noaa.gov/images/aurora-forecast-northern-hemisphere.jpg?t={ts}", 
                 caption="NOAA OVATION MODEL (LIVE)", use_container_width=True)
    
    # 2. Bild: Die Sonne (SDO)
    with tab2:
        # Wir holen das 'AIA 193' Bild (Grün) - zeigt Koronale Löcher
        st.image("https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg", 
                 caption="NASA SDO SATELLITE (AIA 193 - CORONAL HOLES)", use_container_width=True)
        st.caption("ℹ️ Die dunklen Flecken auf der Sonne sind 'Koronale Löcher'. Von dort entweicht der Sonnenwind, der Stürme auslöst.")
# ... (nach den Spalten und Bildern) ...

st.markdown("---")
with st.expander("/// MISSION BRIEFING: PARAMETER GUIDE (CLICK TO EXPAND)"):
    st.markdown("""
    **1. SOLAR WIND SPEED (Geschwindigkeit)**
    * *Normal:* 300-400 km/s.
    * *Sturm:* > 600 km/s.
    * *Effekt:* Je schneller, desto härter trifft das Plasma auf das Erdmagnetfeld.

    **2. ION DENSITY (Dichte)**
    * *Normal:* 5-10 Teilchen/cm³.
    * *Sturm:* > 20 Teilchen/cm³.
    * *Effekt:* Wie "dick" die Wolke ist. Hohe Dichte = Mehr Druck.

    **3. IMF BZ (Interplanetary Magnetic Field)**
    * *WICHTIGSTER WERT!*
    * *Positiv (+):* Das Erdmagnetfeld blockt den Sonnenwind ab (Schutzschild oben).
    * *Negativ (-):* Das Erdmagnetfeld verbindet sich mit dem Sonnenwind ("Tür steht offen"). Energie strömt ein -> Polarlichter!
    
    **4. KP-INDEX**
    * Skala von 0 bis 9 für geomagnetische Aktivität.
    * *Kp 5:* Kleiner Sturm.
    * *Kp 9:* Extremer Sonnensturm (Polarlichter bis in die Schweiz/Deutschland möglich).
    """)

# FOOTER
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #444; font-size: 11px; border-top: 1px solid #222; padding-top: 20px;'>
    SYSTEM DEVELOPED BY J.P. GERBER<br>
    <br>
    Cheers
</div>
""", unsafe_allow_html=True)