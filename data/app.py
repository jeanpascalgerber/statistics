import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# 1. SETUP & "NASA STEALTH" DESIGN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Aurora Forecast Pro", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp { background-color: #0E1117; color: #C9D1D9; }
    
    /* Hide Streamlit Bloat */
    footer, header, .stDeployButton { visibility: hidden; }
    
    /* NASA HUD Boxen */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #58A6FF;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.2);
        padding: 10px;
    }
    
    /* TITEL: WEISS & GROSS */
    div[data-testid="stMetricLabel"] p {
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 800 !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #58A6FF !important;
        font-size: 38px !important;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(88, 166, 255, 0.6);
    }

    /* Slider & Buttons */
    .stSlider > div > div > div > div { background-color: #00FF80; }
    div.stButton > button {
        background-color: #238636; color: white; border: 1px solid #2EA043; width: 100%; font-weight: bold;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background-color: #0d1117; border: 1px solid #30363d; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #1f6feb; color: white; border-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PROFI-GRAFIK: DER NOAA-STYLE GLOBUS
# -----------------------------------------------------------------------------
def create_ultimate_map(probability):
    # Logik: Wo sitzt der Ring?
    center_lat = 90 - (15 + 20 * probability)  # Je höher prob, desto südlicher das Zentrum
    width = 3 + (10 * probability)             # Je höher prob, desto breiter der Ring

    # Farben & Glow definieren (NOAA Style)
    if probability > 0.8:
        core_color = '#FF00FF' # Violett (Core)
        glow_color = '#8A2BE2' # Lila (Glow)
        view_line_lat = 45     # Bis zur Schweiz sichtbar
        status_text = "EXTREM (Kp 7+)"
    elif probability > 0.5:
        core_color = '#FF4500' # Rot (Core)
        glow_color = '#FF8C00' # Orange (Glow)
        view_line_lat = 50     # Deutschland
        status_text = "STARK (Kp 5-6)"
    else:
        core_color = '#00FF00' # Grün (Core)
        glow_color = '#32CD32' # Hellgrün (Glow)
        view_line_lat = 60     # Skandinavien
        status_text = "MODERAT"

    fig = go.Figure()

    # 1. PARTICLE SYSTEM (Das "Gas-Band")
    # Wir erzeugen 3 Ringe mit Punkten für einen weichen Übergang (Glow Effekt)
    
    # Ring A: Der helle Kern
    lons_a = list(range(-180, 180, 2))
    lats_a = [center_lat] * len(lons_a)
    fig.add_trace(go.Scattergeo(
        lon=lons_a, lat=lats_a, mode='markers',
        marker=dict(size=12, color=core_color, opacity=0.8, symbol='circle'), # Gross & Hell
        hoverinfo='skip', name='Aurora Core'
    ))

    # Ring B: Der äussere Schein (Südlich)
    lats_b = [center_lat - width/2] * len(lons_a)
    fig.add_trace(go.Scattergeo(
        lon=lons_a, lat=lats_b, mode='markers',
        marker=dict(size=8, color=glow_color, opacity=0.4), # Kleiner & Transparenter
        hoverinfo='skip', name='Aurora Glow'
    ))
    
    # Ring C: Der innere Schein (Nördlich)
    lats_c = [center_lat + width/2] * len(lons_a)
    fig.add_trace(go.Scattergeo(
        lon=lons_a, lat=lats_c, mode='markers',
        marker=dict(size=8, color=glow_color, opacity=0.4),
        hoverinfo='skip', name='Aurora Glow'
    ))

    # 2. DIE SICHTLINIE (View Line) - Bis hierhin kann man es sehen
    lons_line = list(range(-180, 181, 5))
    lats_line = [view_line_lat] * len(lons_line)
    fig.add_trace(go.Scattergeo(
        lon=lons_line, lat=lats_line, mode='lines',
        line=dict(width=2, color='red', dash='dash'),
        name='Sichtbarkeitsgrenze'
    ))

    # 3. STANDORT ZÜRICH
    fig.add_trace(go.Scattergeo(
        lon=[8.5], lat=[47.4], text=["Zürich"],
        mode='markers+text', textposition="bottom center",
        marker=dict(size=10, color='white', symbol='star'),
        name='Du bist hier'
    ))

    # 4. LAYOUT: Dunkelgraue Erde (Wie gewünscht)
    fig.update_layout(
        title=dict(text=f'AURORA STATUS: {status_text}', font=dict(color="white", size=20, family="Arial Black")),
        geo=dict(
            projection_type='orthographic',
            showland=True, landcolor='#2A2A2A',     # Dunkelgraues Land (Sichtbar aber dezent)
            showocean=True, oceancolor='#0E1117',   # Ozean = Schwarz
            showcoastlines=True, coastlinecolor='#404040',
            showlakes=False, bgcolor='#0E1117',
            center=dict(lat=55, lon=10),
            projection_rotation=dict(lon=10, lat=30, roll=0)
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='#0E1117',
        showlegend=False,
        height=500
    )
    return fig

# -----------------------------------------------------------------------------
# 3. HELPER: GAUGE CHART (TACHO)
# -----------------------------------------------------------------------------
def create_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Wahrscheinlichkeit", 'font': {'color': "white", 'size': 18}},
        number = {'suffix': "%", 'font': {'color': "white"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#1f6feb"}, # Zeiger Farbe
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 40], 'color': '#003300'},   # Dunkelgrün
                {'range': [40, 70], 'color': '#332200'},  # Dunkelorange
                {'range': [70, 100], 'color': '#330033'}  # Dunkelviolett
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Arial"}, height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# -----------------------------------------------------------------------------
# 4. DATEN & MODELL (Core Logic)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_train_model():
    file_path = 'omni_data.txt'
    col_names = ['Year', 'DOY', 'Hour', 'By', 'Bz', 'Density', 'Speed', 'Kp_10']
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, names=col_names)
    except FileNotFoundError:
        return None, None, None, None
    
    df = df.replace([999.9, 9999.99, 999.99, 9999, 99.9], np.nan)
    df['Kp'] = df['Kp_10'] / 10.0
    features = ['Bz', 'By', 'Speed', 'Density']
    df_model = df[features + ['Kp']].dropna().copy()
    
    # 3h Lag
    df_model['Is_Storm'] = (df_model['Kp'] >= 5).astype(int)
    df_model['Target_Future'] = df_model['Is_Storm'].shift(-3)
    df_model = df_model.dropna()
    
    X = df_model[features]
    y = df_model['Target_Future']
    
    model = LogisticRegression(class_weight='balanced', solver='liblinear')
    model.fit(X, y)
    score = model.score(X, y)
    
    return model, df_model, score, features

model, df_hist, score, feature_names = load_and_train_model()

if model is None:
    st.error("❌ 'omni_data.txt' fehlt!")
    st.stop()

# Live Daten holen
last_row = df_hist.iloc[-1]
default_vals = {
    'bz': float(last_row['Bz']),
    'speed': float(last_row['Speed']),
    'density': float(last_row['Density']),
    'by': float(last_row['By'])
}

# -----------------------------------------------------------------------------
# 5. UI: SIDEBAR & RESET
# -----------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a4/Aurora_Borealis_from_the_ISS.jpg")
st.sidebar.markdown("### 🎛️ Mission Control")

if 'bz_val' not in st.session_state: st.session_state.bz_val = default_vals['bz']
if 'speed_val' not in st.session_state: st.session_state.speed_val = default_vals['speed']
if 'density_val' not in st.session_state: st.session_state.density_val = default_vals['density']
if 'by_val' not in st.session_state: st.session_state.by_val = default_vals['by']

def reset_values():
    st.session_state.bz_val = default_vals['bz']
    st.session_state.speed_val = default_vals['speed']
    st.session_state.density_val = default_vals['density']
    st.session_state.by_val = default_vals['by']

if st.sidebar.button("📡 RESET AUF ECHTZEIT-DATEN"):
    reset_values()

st.sidebar.markdown("---")
input_bz = st.sidebar.slider("Magnetfeld Bz (nT)", -30.0, 30.0, key='bz_val')
input_speed = st.sidebar.slider("Speed (km/s)", 200.0, 1200.0, key='speed_val')
input_density = st.sidebar.slider("Density (p/cm³)", 0.1, 80.0, key='density_val')
input_by = st.sidebar.slider("Magnetfeld By (nT)", -30.0, 30.0, key='by_val')

# Berechnung
input_data = pd.DataFrame([[input_bz, input_by, input_speed, input_density]], columns=feature_names)
prob = model.predict_proba(input_data)[0, 1]

# -----------------------------------------------------------------------------
# 6. UI: MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.markdown("## Aurora Forecast Pro <span style='font-size:15px; color:gray'>| Real-time Space Weather Analytics</span>", unsafe_allow_html=True)

# Top KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Modell Genauigkeit", f"{score:.1%}")
col2.metric("Vorhersage Horizont", "+ 3 Stunden")
if prob > 0.7:
    col3.metric("Status", "ALARM STUFE ROT", delta="KRITISCH")
elif prob > 0.5:
    col3.metric("Status", "HOCH", delta="WARNUNG", delta_color="off")
else:
    col3.metric("Status", "RUHIG", delta="NORMAL")

st.markdown("---")

# Split Layout: Gauge Links, Map Rechts
c_left, c_right = st.columns([1, 2])

with c_left:
    st.markdown("#### ⚡ Risiko-Analyse")
    # Der neue TACHO
    fig_gauge = create_gauge(prob)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Text-Fazit
    if prob > 0.8:
        st.error("### 🟣 EXTREMER STURM!")
        st.markdown("**Empfehlung:** Geh raus! Selbst in Zürich könnten Polarlichter sichtbar sein. Kamera bereitmachen.")
    elif prob > 0.5:
        st.warning("### 🟠 AKTIVITÄT HOCH")
        st.markdown("**Empfehlung:** In Norddeutschland/Skandinavien gute Chancen. Webbcam checken!")
    else:
        st.success("### 🟢 RUHIG")
        st.markdown("**Empfehlung:** Schlaf weiter. Die Sonne ist entspannt.")

with c_right:
    st.markdown("#### 🌍 Sichtbarkeits-Prognose")
    # Die neue, krassere Map
    fig_map = create_ultimate_map(prob)
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("ℹ️ **Legende:** Die bunten Punkte zeigen das Plasma-Band. Die rote Linie zeigt, wie weit südlich man es am Horizont sehen kann.")

# -----------------------------------------------------------------------------
# 7. ANALYSE BEREICH (Tabs)
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📊 Why? (Modell-Erklärung)", "📍 Wo stehe ich? (Daten)"])

with tab1:
    st.markdown("#### Feature Importance: Was treibt den Sturm?")
    coeffs = pd.DataFrame({'Feature': feature_names, 'Gewicht': model.coef_[0]}).sort_values(by='Gewicht')
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    fig2.patch.set_facecolor('#0E1117')
    ax2.set_facecolor('#0E1117')
    colors = ['#58A6FF' if c < 0 else '#FF4B4B' for c in coeffs['Gewicht']]
    sns.barplot(x='Gewicht', y='Feature', data=coeffs, palette=colors, ax=ax2)
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    st.pyplot(fig2)
    st.info("Info: Ein negatives Bz (blauer Balken) ist der stärkste Treiber für Stürme, da es das Erdmagnetfeld 'öffnet'.")

with tab2:
    st.markdown("#### Vergleich mit historischen Stürmen")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    sns.scatterplot(x='Bz', y='Speed', data=df_hist.sample(2000), color='#2F363D', alpha=0.3, s=30, ax=ax, edgecolor=None)
    # Highlight Current
    ax.scatter(input_bz, input_speed, color='#FF00FF', s=300, marker='*', label='AKTUELLE WERTE', zorder=10)
    # Style
    ax.set_xlabel("Magnetfeld Bz (nT)", color='white')
    ax.set_ylabel("Speed (km/s)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#161B22', labelcolor='white')
    ax.grid(color='#30363D', linestyle='--', alpha=0.5)
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# 8. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8B949E; font-family: monospace; opacity: 0.7;'>
    Statistik Projekt<br>
    Built by <span style='color: white; font-weight: bold;'>J.P. Gerber</span> | Data: NASA OMNIWeb
</div>
""", unsafe_allow_html=True)