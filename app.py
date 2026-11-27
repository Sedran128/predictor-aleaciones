import os
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Simulador MPEA v2.0 | Materiales 2",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ESTILOS CSS: TEMA AZUL INGENIERÍA PROFESIONAL ---
st.markdown(
    """
<style>
/* Fondo general y barra lateral en paleta azul profundo */
.stApp {
    background: radial-gradient(circle at top right, #0f2547 0%, #081224 55%, #050c18 100%);
    color: #f0f6ff;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1930 0%, #081326 100%);
    border-right: 1px solid #1e3a5f;
}
header[data-testid="stHeader"] {
    background: rgba(8, 18, 36, 0.85);
    backdrop-filter: blur(8px);
}

/* Encabezado principal estilo banner técnico */
.hero-banner {
    background: linear-gradient(135deg, #0f274c 0%, #163b72 52%, #0d2242 100%);
    border: 1px solid #29528a;
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(3, 10, 24, 0.45);
}
.hero-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}
.hero-pill {
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(96, 165, 250, 0.35);
    color: #dbeafe;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.82rem;
    font-weight: 500;
}
.hero-pill strong {
    color: #7dd3fc;
}

/* Tarjeta de estructura cristalina SVG */
.crystal-card {
    background: linear-gradient(155deg, #0d2140 0%, #14325e 55%, #0a1830 100%);
    border: 1px solid #2b548c;
    border-radius: 16px;
    padding: 18px 14px;
    text-align: center;
    color: #f0f6ff;
    box-shadow: 0 8px 22px rgba(4, 12, 28, 0.45);
}
.crystal-title {
    font-size: 1.12rem;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 4px;
    color: #e0f2fe;
}
.crystal-sub {
    font-size: 0.82rem;
    color: #93c5fd;
    margin-bottom: 0;
}

/* Títulos de sección */
.sec-heading {
    font-size: 1.08rem;
    font-weight: 700;
    color: #bae6fd;
    letter-spacing: 0.01em;
    margin: 4px 0 8px 0;
    padding-bottom: 5px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.25);
}

/* Cuadrícula responsiva de KPIs (sin recortes de números ni unidades) */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(162px, 1fr));
    gap: 10px;
    margin-top: 6px;
    margin-bottom: 16px;
}
.kpi-card {
    background: linear-gradient(160deg, rgba(18, 41, 78, 0.92) 0%, rgba(11, 26, 51, 0.95) 100%);
    border: 1px solid #1e3f6e;
    border-top: 2px solid #2563eb;
    border-radius: 12px;
    padding: 11px 13px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    cursor: help;
    transition: border-color 0.2s ease, transform 0.15s ease;
}
.kpi-card:hover {
    border-color: #38bdf8;
    transform: translateY(-1px);
}
.kpi-card.hr-accent {
    border-top-color: #0ea5e9;
    background: linear-gradient(160deg, rgba(14, 39, 71, 0.92) 0%, rgba(9, 24, 46, 0.95) 100%);
}
.kpi-label {
    font-size: 0.78rem;
    color: #93b4d8;
    font-weight: 600;
    line-height: 1.25;
    margin-bottom: 6px;
}
.kpi-value-row {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 5px;
}
.kpi-num {
    font-size: 1.48rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    letter-spacing: -0.01em;
}
.kpi-unit {
    font-size: 0.85rem;
    font-weight: 700;
    color: #38bdf8;
    white-space: nowrap;
}
.kpi-hint {
    font-size: 0.72rem;
    color: #7da2cc;
    margin-top: 5px;
    line-height: 1.2;
}

/* Tablas descriptivas de referencia */
.info-table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 20px 0;
    font-size: 0.86rem;
    background: rgba(11, 26, 51, 0.68);
    border: 1px solid #1e3f6e;
    border-radius: 10px;
    overflow: hidden;
}
.info-table th {
    background: rgba(22, 59, 114, 0.72);
    color: #bae6fd;
    text-align: left;
    padding: 10px 12px;
    font-weight: 700;
    border-bottom: 1px solid #29528a;
}
.info-table td {
    padding: 9px 12px;
    color: #e2e8f0;
    border-bottom: 1px solid rgba(56, 189, 248, 0.14);
    vertical-align: top;
    line-height: 1.4;
}
.info-table tr:last-child td {
    border-bottom: none;
}
.info-table tr:hover td {
    background: rgba(56, 189, 248, 0.06);
}
.info-badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(96, 165, 250, 0.35);
    color: #7dd3fc;
    border-radius: 6px;
    padding: 1px 7px;
    font-weight: 600;
    font-size: 0.8rem;
    white-space: nowrap;
}

/* Barras de probabilidad, botones y pestañas en azul cobalto */
div[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #2563eb 0%, #38bdf8 100%);
}
button[data-baseweb="tab"] {
    color: #93b4d8;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}
div[data-testid="stSidebar"] button[kind="secondary"] {
    background-color: #132d54;
    border: 1px solid #29528a;
    color: #e0f2fe;
}
div[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background-color: #1d4ed8;
    border-color: #60a5fa;
    color: #ffffff;
}
hr {
    border-color: rgba(56, 189, 248, 0.2) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def render_kpi_grid(items, extra_class=""):
    cards = []
    cls = f"kpi-card {extra_class}".strip()
    for item in items:
        label, valor, unidad, hint = item[:4]
        tooltip = item[4] if len(item) > 4 else ""
        title_attr = f' title="{tooltip}"' if tooltip else ""
        hint_html = f'<div class="kpi-hint">{hint}</div>' if hint else ""
        unit_html = f'<span class="kpi-unit">{unidad}</span>' if unidad else ""
        cards.append(
            f'<div class="{cls}"{title_attr}>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value-row"><span class="kpi-num">{valor}</span>{unit_html}</div>'
            f'{hint_html}'
            f'</div>'
        )
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def render_info_table(headers, rows):
    th_html = "".join(f"<th>{h}</th>" for h in headers)
    tr_list = []
    for row in rows:
        tds = []
        for idx, cell in enumerate(row):
            if idx == 1 and len(headers) == 4:
                tds.append(f'<td><span class="info-badge">{cell}</span></td>')
            elif idx == 0:
                tds.append(f"<td><strong>{cell}</strong></td>")
            else:
                tds.append(f"<td>{cell}</td>")
        tr_list.append(f"<tr>{''.join(tds)}</tr>")
    return (
        f'<table class="info-table">'
        f'<thead><tr>{th_html}</tr></thead>'
        f'<tbody>{"".join(tr_list)}</tbody>'
        f'</table>'
    )


def estilizar_plotly(fig, height=450):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(11, 26, 51, 0.55)",
        plot_bgcolor="rgba(8, 18, 36, 0.72)",
        font=dict(color="#e2e8f0", family="Inter, Segoe UI, sans-serif"),
        title_font=dict(color="#bae6fd", size=15),
        legend=dict(bgcolor="rgba(11, 26, 51, 0.7)", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=45, r=25, t=55, b=45),
    )
    fig.update_xaxes(gridcolor="rgba(56, 189, 248, 0.12)", zerolinecolor="rgba(56, 189, 248, 0.25)")
    fig.update_yaxes(gridcolor="rgba(56, 189, 248, 0.12)", zerolinecolor="rgba(56, 189, 248, 0.25)")
    return fig


# --- CARGA DE MODELO v2.0 Y REFERENCIA ASHBY ---
@st.cache_resource
def cargar_modelos():
    return joblib.load("modelos_mpea.joblib")


@st.cache_data
def cargar_dataset_referencia():
    if os.path.exists("mpea_fusionado_ref.csv"):
        df = pd.read_csv("mpea_fusionado_ref.csv")
        for c in ["ys", "elong", "dens"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["ys", "phase"])
    if os.path.exists("MPEA_dataset.csv"):
        df = pd.read_csv("MPEA_dataset.csv")
        rename_map = {
            "PROPERTY: BCC/FCC/other": "phase",
            "PROPERTY: YS (MPa)": "ys",
            "PROPERTY: Elongation (%)": "elong",
            r"PROPERTY: Calculated Density (g/cm$^3$)": "dens",
        }
        df = df.rename(columns=rename_map)
        for c in ["ys", "elong", "dens"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["source"] = "MPEA_Borg"
        cols = [c for c in ["FORMULA", "phase", "ys", "elong", "dens", "source"] if c in df.columns]
        return df[cols].dropna(subset=["ys", "phase"])
    return None


datos = cargar_modelos()
df_ref = cargar_dataset_referencia()

if datos.get("version") != "2.0":
    st.error("⚠️ El archivo `modelos_mpea.joblib` en el repositorio es anterior a v2.0. Sube el nuevo archivo y reinicia la app en Streamlit.")
    st.stop()

rf_cls = datos["rf_cls"]
rf_ys = datos["rf_ys"]
rf_uts = datos["rf_uts"]
rf_elong = datos["rf_elong"]
rf_hv = datos["rf_hv"]
rf_dens = datos["rf_dens"]
rf_young = datos["rf_young"]

elementos_modelo = sorted(datos["elementos"])
procesamientos = datos["procesamientos"]
ELEM_PROPS = datos["elem_props"]
HR_COLS = datos["hr_cols"]
cols_cls = datos["cols_cls"]
cols_ys = datos["cols_ys"]
cols_hv = datos["cols_hv"]
cols_dens = datos["cols_dens"]
cols_young = datos["cols_young"]

NOMBRES_PROC = {
    "CAST": "Fundición",
    "ANNEAL": "Recocido",
    "WROUGHT": "Forjado / Laminado",
    "POWDER": "Pulvimetalurgia",
    "OTHER": "Otro proceso",
}
INV_PROC = {v: k for k, v in NOMBRES_PROC.items()}

COLORES_FASE = {
    "FCC": "#3b82f6",
    "BCC": "#f97316",
    "HCP": "#10b981",
    "FCC + BCC": "#38bdf8",
    "SS + Intermetálicos": "#a855f7",
    "Amorfa": "#f472b6",
}

PRESETS = {
    "Cantor — CoCrFeMnNi (FCC)": {"Co": 20.0, "Cr": 20.0, "Fe": 20.0, "Mn": 20.0, "Ni": 20.0},
    "Refractaria — NbTaTiVZr (BCC)": {"Nb": 20.0, "Ta": 20.0, "Ti": 20.0, "V": 20.0, "Zr": 20.0},
    "Dual — Al0.5CoCrFeNi (FCC + BCC)": {"Al": 11.1, "Co": 22.2, "Cr": 22.2, "Fe": 22.2, "Ni": 22.3},
    "Alta Resistencia — AlCoFeNi (BCC / B2)": {"Al": 25.0, "Co": 25.0, "Fe": 25.0, "Ni": 25.0},
    "Hexagonal — TiZrHfScY (HCP)": {"Ti": 20.0, "Zr": 20.0, "Hf": 20.0, "Sc": 20.0, "Y": 20.0},
    "Vidrio Metálico — Zr55Cu30Al10Ni5 (Amorfa)": {"Zr": 55.0, "Cu": 30.0, "Al": 10.0, "Ni": 5.0},
}


# --- FUNCIONES FÍSICAS Y DE PREDICCIÓN ---
def parse_formula(formula):
    matches = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    comp = {}
    for el, qty in matches:
        if el in ELEM_PROPS:
            val = float(qty) if qty != "" else 1.0
            comp[el] = comp.get(el, 0.0) + val
    total = sum(comp.values())
    if total > 0:
        comp = {k: v / total for k, v in comp.items()}
    return comp


def calc_hr(comp):
    c = {el: v for el, v in comp.items() if el in ELEM_PROPS and v > 0}
    tot = sum(c.values())
    if tot <= 0:
        return {k: 0.0 for k in HR_COLS}
    c = {el: v / tot for el, v in c.items()}
    vec = sum(f * ELEM_PROPS[el][0] for el, f in c.items())
    r_bar = sum(f * ELEM_PROPS[el][1] for el, f in c.items())
    chi_bar = sum(f * ELEM_PROPS[el][2] for el, f in c.items())
    tm = sum(f * ELEM_PROPS[el][3] for el, f in c.items())
    delta = 100.0 * np.sqrt(sum(f * (1.0 - ELEM_PROPS[el][1] / r_bar) ** 2 for el, f in c.items()))
    d_chi = np.sqrt(sum(f * (ELEM_PROPS[el][2] - chi_bar) ** 2 for el, f in c.items()))
    dSmix = -8.314462618 * sum(f * np.log(f) for f in c.values() if f > 0)
    return {
        "HR_VEC": float(vec),
        "HR_delta": float(delta),
        "HR_dchi": float(d_chi),
        "HR_Tm": float(tm),
        "HR_dSmix": float(dSmix),
        "HR_r_mean": float(r_bar),
    }


def construir_fila(comp_dict, columnas, temp=None, proc_code=None):
    hr = calc_hr(comp_dict)
    fila = {col: 0.0 for col in columnas}
    for el, val in comp_dict.items():
        if el in fila:
            fila[el] = val
    for k, v in hr.items():
        if k in fila:
            fila[k] = v
    if temp is not None:
        if "TEMP_C" in fila:
            fila["TEMP_C"] = float(temp)
        if "HR_T_homologa" in fila:
            fila["HR_T_homologa"] = (float(temp) + 273.15) / hr["HR_Tm"] if hr["HR_Tm"] > 0 else 0.2
    if proc_code is not None:
        col_p = f"PROC_{proc_code}"
        if col_p in fila:
            fila[col_p] = 1.0
    return pd.DataFrame([fila])[columnas]


def predecir_todo(comp_dict, temp, proc_code):
    hr = calc_hr(comp_dict)
    X_cls = construir_fila(comp_dict, cols_cls)
    X_tp = construir_fila(comp_dict, cols_ys, temp=temp, proc_code=proc_code)
    X_hv = construir_fila(comp_dict, cols_hv, proc_code=proc_code)
    X_dens = construir_fila(comp_dict, cols_dens)
    X_young = construir_fila(comp_dict, cols_young)

    fase = rf_cls.predict(X_cls)[0]
    probs = dict(zip(rf_cls.classes_, rf_cls.predict_proba(X_cls)[0]))
    ys = float(rf_ys.predict(X_tp)[0])
    uts = float(rf_uts.predict(X_tp)[0])
    elong = float(rf_elong.predict(X_tp)[0])
    hv = float(rf_hv.predict(X_hv)[0])
    dens = float(rf_dens.predict(X_dens)[0])
    young = float(rf_young.predict(X_young)[0])
    ys_esp = ys / dens if dens > 0 else 0.0
    t_hom = (float(temp) + 273.15) / hr["HR_Tm"] if hr["HR_Tm"] > 0 else 0.0

    return {
        "fase": fase,
        "probs": probs,
        "ys": ys,
        "uts": uts,
        "elong": elong,
        "hv": hv,
        "dens": dens,
        "young": young,
        "ys_esp": ys_esp,
        "hr": hr,
        "t_hom": t_hom,
    }


def svg_estructura_cristalina(fase):
    if fase == "FCC":
        svg = """
<svg viewBox="0 0 200 200" width="150" height="150" xmlns="http://www.w3.org/2000/svg">
<defs>
<radialGradient id="gCornerF" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#93c5fd"/><stop offset="100%" stop-color="#2563eb"/>
</radialGradient>
<radialGradient id="gFaceF" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#67e8f9"/><stop offset="100%" stop-color="#0891b2"/>
</radialGradient>
</defs>
<g stroke="#64748b" stroke-width="1.6" stroke-dasharray="4,3" fill="none">
<line x1="75" y1="35" x2="75" y2="125"/>
<line x1="75" y1="125" x2="165" y2="125"/>
<line x1="75" y1="125" x2="35" y2="165"/>
</g>
<g stroke="#cbd5e1" stroke-width="2" fill="none">
<rect x="35" y="75" width="90" height="90"/>
<line x1="35" y1="75" x2="75" y2="35"/>
<line x1="125" y1="75" x2="165" y2="35"/>
<line x1="125" y1="165" x2="165" y2="125"/>
<line x1="75" y1="35" x2="165" y2="35"/>
<line x1="165" y1="35" x2="165" y2="125"/>
</g>
<circle cx="75" cy="125" r="9.5" fill="url(#gCornerF)" opacity="0.7"/>
<circle cx="120" cy="80" r="10.5" fill="url(#gFaceF)" opacity="0.8"/>
<circle cx="100" cy="55" r="11" fill="url(#gFaceF)"/>
<circle cx="100" cy="145" r="11" fill="url(#gFaceF)"/>
<circle cx="55" cy="100" r="11" fill="url(#gFaceF)"/>
<circle cx="145" cy="100" r="11" fill="url(#gFaceF)"/>
<circle cx="75" cy="35" r="10" fill="url(#gCornerF)"/>
<circle cx="165" cy="35" r="10" fill="url(#gCornerF)"/>
<circle cx="165" cy="125" r="10" fill="url(#gCornerF)"/>
<circle cx="35" cy="75" r="10.5" fill="url(#gCornerF)"/>
<circle cx="125" cy="75" r="10.5" fill="url(#gCornerF)"/>
<circle cx="35" cy="165" r="10.5" fill="url(#gCornerF)"/>
<circle cx="125" cy="165" r="10.5" fill="url(#gCornerF)"/>
<circle cx="80" cy="120" r="11.5" fill="url(#gFaceF)" stroke="#e0f2fe" stroke-width="1.2"/>
</svg>
"""
        titulo = "Estructura FCC"
        sub = "Cúbica Centrada en las Caras · Alta ductilidad y tenacidad"
    elif fase == "BCC":
        svg = """
<svg viewBox="0 0 200 200" width="150" height="150" xmlns="http://www.w3.org/2000/svg">
<defs>
<radialGradient id="gCornerB" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#fdba74"/><stop offset="100%" stop-color="#ea580c"/>
</radialGradient>
<radialGradient id="gCenterB" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#fef08a"/><stop offset="100%" stop-color="#dc2626"/>
</radialGradient>
</defs>
<g stroke="#64748b" stroke-width="1.6" stroke-dasharray="4,3" fill="none">
<line x1="75" y1="35" x2="75" y2="125"/>
<line x1="75" y1="125" x2="165" y2="125"/>
<line x1="75" y1="125" x2="35" y2="165"/>
</g>
<g stroke="#f97316" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.65">
<line x1="35" y1="75" x2="165" y2="125"/>
<line x1="125" y1="165" x2="75" y2="35"/>
<line x1="125" y1="75" x2="75" y2="125"/>
<line x1="35" y1="165" x2="165" y2="35"/>
</g>
<g stroke="#cbd5e1" stroke-width="2" fill="none">
<rect x="35" y="75" width="90" height="90"/>
<line x1="35" y1="75" x2="75" y2="35"/>
<line x1="125" y1="75" x2="165" y2="35"/>
<line x1="125" y1="165" x2="165" y2="125"/>
<line x1="75" y1="35" x2="165" y2="35"/>
<line x1="165" y1="35" x2="165" y2="125"/>
</g>
<circle cx="75" cy="125" r="10" fill="url(#gCornerB)" opacity="0.7"/>
<circle cx="100" cy="100" r="14" fill="url(#gCenterB)" stroke="#fef9c3" stroke-width="1.5"/>
<circle cx="75" cy="35" r="10.5" fill="url(#gCornerB)"/>
<circle cx="165" cy="35" r="10.5" fill="url(#gCornerB)"/>
<circle cx="165" cy="125" r="10.5" fill="url(#gCornerB)"/>
<circle cx="35" cy="75" r="11" fill="url(#gCornerB)"/>
<circle cx="125" cy="75" r="11" fill="url(#gCornerB)"/>
<circle cx="35" cy="165" r="11" fill="url(#gCornerB)"/>
<circle cx="125" cy="165" r="11" fill="url(#gCornerB)"/>
</svg>
"""
        titulo = "Estructura BCC"
        sub = "Cúbica Centrada en el Cuerpo · Alta resistencia y refractariedad"
    elif fase == "HCP":
        svg = """
<svg viewBox="0 0 200 200" width="150" height="150" xmlns="http://www.w3.org/2000/svg">
<defs>
<radialGradient id="gHcpOut" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#6ee7b7"/><stop offset="100%" stop-color="#059669"/>
</radialGradient>
<radialGradient id="gHcpMid" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#a7f3d0"/><stop offset="100%" stop-color="#0d9488"/>
</radialGradient>
</defs>
<g stroke="#64748b" stroke-width="1.5" stroke-dasharray="4,3" fill="none">
<polygon points="70,128 130,128 160,150 130,172 70,172 40,150"/>
<line x1="70" y1="28" x2="70" y2="128"/>
<line x1="130" y1="28" x2="130" y2="128"/>
</g>
<g stroke="#cbd5e1" stroke-width="2" fill="none">
<polygon points="70,28 130,28 160,50 130,72 70,72 40,50"/>
<polyline points="40,150 70,172 130,172 160,150"/>
<line x1="40" y1="50" x2="40" y2="150"/>
<line x1="70" y1="72" x2="70" y2="172"/>
<line x1="130" y1="72" x2="130" y2="172"/>
<line x1="160" y1="50" x2="160" y2="150"/>
</g>
<polygon points="78,94 122,94 100,114" fill="rgba(16,185,129,0.22)" stroke="#34d399" stroke-width="1.3"/>
<circle cx="70" cy="128" r="8.5" fill="url(#gHcpOut)" opacity="0.65"/>
<circle cx="130" cy="128" r="8.5" fill="url(#gHcpOut)" opacity="0.65"/>
<circle cx="100" cy="150" r="9.5" fill="url(#gHcpMid)" opacity="0.8"/>
<circle cx="78" cy="94" r="10.5" fill="url(#gHcpMid)" stroke="#d1fae5" stroke-width="1.2"/>
<circle cx="122" cy="94" r="10.5" fill="url(#gHcpMid)" stroke="#d1fae5" stroke-width="1.2"/>
<circle cx="100" cy="114" r="10.5" fill="url(#gHcpMid)" stroke="#d1fae5" stroke-width="1.2"/>
<circle cx="40" cy="150" r="9" fill="url(#gHcpOut)"/>
<circle cx="70" cy="172" r="9.5" fill="url(#gHcpOut)"/>
<circle cx="130" cy="172" r="9.5" fill="url(#gHcpOut)"/>
<circle cx="160" cy="150" r="9" fill="url(#gHcpOut)"/>
<circle cx="70" cy="28" r="9" fill="url(#gHcpOut)"/>
<circle cx="130" cy="28" r="9" fill="url(#gHcpOut)"/>
<circle cx="40" cy="50" r="9.5" fill="url(#gHcpOut)"/>
<circle cx="160" cy="50" r="9.5" fill="url(#gHcpOut)"/>
<circle cx="70" cy="72" r="9.5" fill="url(#gHcpOut)"/>
<circle cx="130" cy="72" r="9.5" fill="url(#gHcpOut)"/>
<circle cx="100" cy="50" r="10" fill="url(#gHcpMid)" stroke="#d1fae5" stroke-width="1"/>
</svg>
"""
        titulo = "Estructura HCP"
        sub = "Hexagonal Compacta · Típica de tierras raras, Ti, Zr y Hf"
    elif fase == "FCC + BCC":
        svg = """
<svg viewBox="0 0 200 200" width="150" height="150" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="gSplitF" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#3b82f6" stop-opacity="0.32"/><stop offset="100%" stop-color="#06b6d4" stop-opacity="0.14"/>
</linearGradient>
<linearGradient id="gSplitB" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#f97316" stop-opacity="0.35"/><stop offset="100%" stop-color="#ef4444" stop-opacity="0.15"/>
</linearGradient>
</defs>
<rect x="14" y="34" width="80" height="132" rx="10" fill="url(#gSplitF)" stroke="#38bdf8" stroke-width="1.8"/>
<rect x="106" y="34" width="80" height="132" rx="10" fill="url(#gSplitB)" stroke="#fb923c" stroke-width="1.8"/>
<rect x="30" y="72" width="48" height="48" fill="none" stroke="#bae6fd" stroke-width="1.8"/>
<circle cx="30" cy="72" r="5.5" fill="#60a5fa"/><circle cx="78" cy="72" r="5.5" fill="#60a5fa"/>
<circle cx="30" cy="120" r="5.5" fill="#60a5fa"/><circle cx="78" cy="120" r="5.5" fill="#60a5fa"/>
<circle cx="54" cy="96" r="7" fill="#67e8f9" stroke="#e0f2fe" stroke-width="1"/>
<text x="54" y="152" text-anchor="middle" fill="#bae6fd" font-size="13" font-family="sans-serif" font-weight="bold">FCC</text>
<rect x="122" y="72" width="48" height="48" fill="none" stroke="#fed7aa" stroke-width="1.8"/>
<line x1="122" y1="72" x2="170" y2="120" stroke="#fdba74" stroke-width="1" stroke-dasharray="3,2"/>
<line x1="170" y1="72" x2="122" y2="120" stroke="#fdba74" stroke-width="1" stroke-dasharray="3,2"/>
<circle cx="122" cy="72" r="5.5" fill="#fb923c"/><circle cx="170" cy="72" r="5.5" fill="#fb923c"/>
<circle cx="122" cy="120" r="5.5" fill="#fb923c"/><circle cx="170" cy="120" r="5.5" fill="#fb923c"/>
<circle cx="146" cy="96" r="8" fill="#fef08a" stroke="#dc2626" stroke-width="1.5"/>
<text x="146" y="152" text-anchor="middle" fill="#fed7aa" font-size="13" font-family="sans-serif" font-weight="bold">BCC</text>
<circle cx="100" cy="100" r="13" fill="#081224" stroke="#e2e8f0" stroke-width="1.5"/>
<text x="100" y="105" text-anchor="middle" fill="#f8fafc" font-size="16" font-family="sans-serif" font-weight="bold">+</text>
</svg>
"""
        titulo = "Fase Dual (FCC + BCC)"
        sub = "Balance sinérgico entre ductilidad (FCC) y resistencia (BCC)"
    elif fase == "Amorfa":
        svg = """
<svg viewBox="0 0 200 200" width="150" height="150" xmlns="http://www.w3.org/2000/svg">
<defs>
<radialGradient id="gAm1" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#f9a8d4"/><stop offset="100%" stop-color="#db2777"/>
</radialGradient>
<radialGradient id="gAm2" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#c4b5fd"/><stop offset="100%" stop-color="#7c3aed"/>
</radialGradient>
<radialGradient id="gAm3" cx="35%" cy="35%" r="65%">
<stop offset="0%" stop-color="#67e8f9"/><stop offset="100%" stop-color="#0284c7"/>
</radialGradient>
</defs>
<circle cx="100" cy="100" r="82" fill="#081224" stroke="#f472b6" stroke-width="2.2" stroke-dasharray="6,4"/>
<g stroke="#475569" stroke-width="1.2" opacity="0.6">
<line x1="60" y1="62" x2="96" y2="54"/><line x1="96" y1="54" x2="136" y2="70"/>
<line x1="60" y1="62" x2="54" y2="105"/><line x1="54" y1="105" x2="88" y2="102"/>
<line x1="88" y1="102" x2="128" y2="108"/><line x1="128" y1="108" x2="150" y2="82"/>
<line x1="54" y1="105" x2="72" y2="145"/><line x1="72" y1="145" x2="112" y2="148"/>
<line x1="112" y1="148" x2="144" y2="132"/><line x1="88" y1="102" x2="112" y2="148"/>
</g>
<circle cx="60" cy="62" r="13" fill="url(#gAm1)"/>
<circle cx="96" cy="54" r="9.5" fill="url(#gAm3)"/>
<circle cx="136" cy="70" r="12" fill="url(#gAm2)"/>
<circle cx="54" cy="105" r="11" fill="url(#gAm2)"/>
<circle cx="88" cy="102" r="14.5" fill="url(#gAm1)" stroke="#fbcfe8" stroke-width="1.2"/>
<circle cx="128" cy="108" r="10" fill="url(#gAm3)"/>
<circle cx="150" cy="82" r="8.5" fill="url(#gAm1)"/>
<circle cx="72" cy="145" r="12.5" fill="url(#gAm3)"/>
<circle cx="112" cy="148" r="11.5" fill="url(#gAm2)"/>
<circle cx="144" cy="132" r="9" fill="url(#gAm1)"/>
<circle cx="105" cy="78" r="7.5" fill="url(#gAm2)"/>
<circle cx="72" cy="82" r="7" fill="url(#gAm3)"/>
</svg>
"""
        titulo = "Estructura Amorfa"
        sub = "Vidrio metálico sin orden periódico de largo alcance (alta δ)"
    else:
        svg = """
<svg viewBox="0 0 200 200" width="150" height="150" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="gDual1" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#3b82f6" stop-opacity="0.32"/>
<stop offset="100%" stop-color="#06b6d4" stop-opacity="0.14"/>
</linearGradient>
<linearGradient id="gDual2" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#f97316" stop-opacity="0.38"/>
<stop offset="100%" stop-color="#a855f7" stop-opacity="0.22"/>
</linearGradient>
</defs>
<circle cx="100" cy="100" r="82" fill="#081224" stroke="#a855f7" stroke-width="2.5"/>
<path d="M 32 100 Q 65 40 120 42 Q 105 95 95 110 Q 55 125 32 100 Z" fill="url(#gDual1)" stroke="#60a5fa" stroke-width="1.5"/>
<path d="M 120 42 Q 170 65 168 120 Q 125 135 95 110 Q 105 95 120 42 Z" fill="url(#gDual2)" stroke="#fb923c" stroke-width="1.5"/>
<path d="M 32 100 Q 55 125 95 110 Q 125 135 168 120 Q 130 175 65 165 Q 35 140 32 100 Z" fill="#a855f7" fill-opacity="0.24" stroke="#c084fc" stroke-width="1.5"/>
<rect x="66" y="74" width="12" height="12" rx="2" fill="#e879f9" transform="rotate(20 72 80)"/>
<polygon points="132,74 140,88 124,88" fill="#fef08a"/>
<rect x="108" y="134" width="13" height="13" rx="2" fill="#e879f9" transform="rotate(-15 114 140)"/>
<circle cx="75" cy="138" r="5.5" fill="#93c5fd"/>
<polygon points="144,106 151,118 137,118" fill="#fdba74"/>
<text x="64" y="62" fill="#bfdbfe" font-size="10.5" font-family="sans-serif" font-weight="bold">Matriz SS</text>
<text x="128" y="102" fill="#fef08a" font-size="10" font-family="sans-serif" font-weight="bold">B2 / L1₂</text>
<text x="68" y="160" fill="#e9d5ff" font-size="10" font-family="sans-serif" font-weight="bold">Laves / σ</text>
</svg>
"""
        titulo = "SS + Intermetálicos"
        sub = "Solución sólida con fases ordenadas (B2, L1₂, Laves o σ)"

    html = f"""
<div class="crystal-card">
{svg.strip()}
<div class="crystal-title">{titulo}</div>
<p class="crystal-sub">{sub}</p>
</div>
"""
    return html.strip()


# --- ESTADO SINCRONIZADO (SLIDER <-> CASILLA % AT.) ---
if "saved_preset" not in st.session_state:
    st.session_state["saved_preset"] = "Cantor — CoCrFeMnNi (FCC)"
    inicial = PRESETS[st.session_state["saved_preset"]]
    st.session_state["saved_sels"] = [e for e in inicial if e in elementos_modelo]
    for el, val in inicial.items():
        st.session_state[f"pct_{el}"] = float(val)


def aplicar_preset():
    p_name = st.session_state["preset_actual"]
    st.session_state["saved_preset"] = p_name
    p_dict = PRESETS[p_name]
    els_validos = [e for e in p_dict if e in elementos_modelo]
    st.session_state["saved_sels"] = els_validos
    st.session_state["seleccionados"] = els_validos
    for el in els_validos:
        val = float(p_dict[el])
        st.session_state[f"pct_{el}"] = val
        st.session_state[f"sl_{el}"] = val
        st.session_state[f"num_{el}"] = val


def sync_sels():
    st.session_state["saved_sels"] = list(st.session_state["seleccionados"])


def sync_from_slider(el):
    val = float(st.session_state[f"sl_{el}"])
    st.session_state[f"pct_{el}"] = val
    st.session_state[f"num_{el}"] = val


def sync_from_num(el):
    val = float(st.session_state[f"num_{el}"])
    st.session_state[f"pct_{el}"] = val
    st.session_state[f"sl_{el}"] = val


def hacer_equiatomica():
    sels = st.session_state.get("saved_sels", [])
    if sels:
        val = round(100.0 / len(sels), 2)
        for el in sels:
            st.session_state[f"pct_{el}"] = val
            st.session_state[f"sl_{el}"] = val
            st.session_state[f"num_{el}"] = val


def normalizar_a_100():
    sels = st.session_state.get("saved_sels", [])
    total = sum(st.session_state.get(f"pct_{el}", 0.0) for el in sels)
    if sels and total > 0:
        for el in sels:
            val = round(st.session_state.get(f"pct_{el}", 0.0) * 100.0 / total, 2)
            st.session_state[f"pct_{el}"] = val
            st.session_state[f"sl_{el}"] = val
            st.session_state[f"num_{el}"] = val


# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown("### ⚙️ Diseño de Aleación")
    modo = st.radio(
        "Modo de entrada:",
        ["🎛️ Constructor (% at.)", "⌨️ Escribir Fórmula"],
        horizontal=True,
    )

    comp_final = {}
    formula_mostrar = ""

    if modo == "🎛️ Constructor (% at.)":
        if "preset_actual" not in st.session_state:
            st.session_state["preset_actual"] = st.session_state["saved_preset"]
        if "seleccionados" not in st.session_state:
            st.session_state["seleccionados"] = list(st.session_state["saved_sels"])

        st.selectbox(
            "Aleación de ejemplo (6 estructuras):",
            options=list(PRESETS.keys()),
            key="preset_actual",
            on_change=aplicar_preset,
        )

        seleccionados = st.multiselect(
            "Elementos químicos:",
            options=elementos_modelo,
            key="seleccionados",
            on_change=sync_sels,
        )

        if seleccionados:
            b1, b2 = st.columns(2)
            b1.button("⚖️ Equiatómica", on_click=hacer_equiatomica, use_container_width=True)
            b2.button("📏 Ajustar a 100%", on_click=normalizar_a_100, use_container_width=True)

            st.caption("Ajusta el **porcentaje atómico (% at.)** con la barra o la casilla:")
            valores_pct = {}
            for el in seleccionados:
                if f"pct_{el}" not in st.session_state:
                    st.session_state[f"pct_{el}"] = round(100.0 / len(seleccionados), 1)
                if f"sl_{el}" not in st.session_state:
                    st.session_state[f"sl_{el}"] = float(st.session_state[f"pct_{el}"])
                if f"num_{el}" not in st.session_state:
                    st.session_state[f"num_{el}"] = float(st.session_state[f"pct_{el}"])

                col_sl, col_num = st.columns([2.1, 1.1])
                with col_sl:
                    st.slider(
                        f"{el} (% at.)",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.5,
                        key=f"sl_{el}",
                        on_change=sync_from_slider,
                        args=(el,),
                    )
                with col_num:
                    st.number_input(
                        f"% {el}",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.5,
                        key=f"num_{el}",
                        on_change=sync_from_num,
                        args=(el,),
                        label_visibility="hidden",
                    )
                valores_pct[el] = st.session_state[f"pct_{el}"]

            suma_pct = sum(valores_pct.values())
            if suma_pct <= 0:
                st.error("Asigna un porcentaje mayor a 0%.")
            else:
                if abs(suma_pct - 100.0) <= 0.25:
                    st.success(f"Suma total: **{suma_pct:.1f}% at.**")
                else:
                    st.warning(f"Suma actual: **{suma_pct:.1f}%** (normalizado a 100% al calcular)")
                comp_final = {k: v / suma_pct for k, v in valores_pct.items() if v > 0}
                formula_mostrar = " ".join([f"{k}{round(v * 100, 1)}" for k, v in comp_final.items()])
        else:
            st.warning("Selecciona al menos un elemento.")
    else:
        formula_texto = st.text_input(
            "Fórmula química (proporciones molares o %):",
            value="Al0.5 Co1 Cr1 Fe1 Ni1",
            help="Ejemplos: CoCrFeMnNi, TiZrHfScY, Zr55Cu30Al10Ni5",
        )
        comp_final = parse_formula(formula_texto)
        formula_mostrar = " ".join([f"{k}{round(v * 100, 1)}" for k, v in comp_final.items()])

    st.divider()
    opciones_proc = [NOMBRES_PROC.get(p, p) for p in procesamientos]
    proc_elegido_label = st.selectbox("🏭 Método de procesamiento:", opciones_proc)
    proc_code = INV_PROC.get(proc_elegido_label, proc_elegido_label)

    temperatura = st.slider(
        "🌡️ Temperatura de prueba (°C):",
        min_value=25,
        max_value=1200,
        value=25,
        step=25,
    )

# --- PANEL PRINCIPAL ---
if not comp_final:
    st.markdown(
        '<div class="hero-banner"><div class="hero-title"> Simulador Predictivo de Aleaciones MPEA v2.0</div>'
        '<div style="color:#bfdbfe;font-size:0.92rem;">Selecciona elementos en el panel izquierdo para calcular estructura y propiedades.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown(
    f"""
<div class="hero-banner">
<div class="hero-title"> Simulador Predictivo de Aleaciones MPEA v2.0</div>
<div class="hero-meta">
<span class="hero-pill"><strong>Composición (% at.):</strong> {formula_mostrar}</span>
<span class="hero-pill"><strong>Procesamiento:</strong> {proc_elegido_label}</span>
<span class="hero-pill"><strong>Temperatura:</strong> {temperatura} °C</span>
<span class="hero-pill"><strong>Motor:</strong> 5 bases fusionadas + Hume-Rothery</span>
</div>
</div>
""",
    unsafe_allow_html=True,
)

res = predecir_todo(comp_final, temperatura, proc_code)
hr = res["hr"]
ratio_uts_ys = res["uts"] / res["ys"] if res["ys"] > 0 else 1.0

# --- BLOQUE SUPERIOR: CELDA CRISTALINA SVG + FICHA TÉCNICA RESPONSIVA ---
col_celda, col_ficha = st.columns([1.05, 2.55], gap="large")

with col_celda:
    st.markdown(svg_estructura_cristalina(res["fase"]), unsafe_allow_html=True)
    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-heading">📊 Confianza por Estructura (6 clases)</div>', unsafe_allow_html=True)
    probs_ordenadas = sorted(res["probs"].items(), key=lambda x: -x[1])
    for ph, p_val in probs_ordenadas:
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; font-size:0.83rem; color:#dbeafe; margin-bottom:2px;'>"
            f"<span>{ph}</span><strong style='color:#7dd3fc;'>{p_val*100:.1f}%</strong></div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(float(p_val), 0.0), 1.0))

with col_ficha:
    st.markdown('<div class="sec-heading">📋 Ficha Técnica Estimada</div>', unsafe_allow_html=True)
    st.caption("Pasa el cursor sobre cualquier tarjeta para ver su definición o consulta la pestaña **📖 Guía de Propiedades**.")
    kpis_mecanicos = [
        (
            "Límite Elástico (YS)",
            f"{res['ys']:.1f}",
            "MPa",
            f"A {temperatura} °C ({proc_elegido_label})",
            "Esfuerzo máximo que soporta la aleación antes de deformarse de forma permanente (fluencia).",
        ),
        (
            "Resistencia Máxima (UTS)",
            f"{res['uts']:.1f}",
            "MPa",
            "Esfuerzo último de tracción",
            "Esfuerzo máximo de ingeniería alcanzado durante un ensayo de tracción antes de la estricción o fractura.",
        ),
        (
            "Elongación / Ductilidad",
            f"{res['elong']:.1f}",
            "%",
            "Deformación hasta fractura",
            "Porcentaje de alargamiento plástico que admite la aleación antes de romperse; indica qué tan dúctil o frágil es.",
        ),
        (
            "Dureza Vickers (HV)",
            f"{res['hv']:.0f}",
            "HV",
            "Estimada a temp. ambiente",
            "Resistencia superficial a la penetración e indentación plástica; suele correlacionar con resistencia al desgaste.",
        ),
        (
            "Densidad (ρ)",
            f"{res['dens']:.2f}",
            "g/cm³",
            "Basada en composición",
            "Masa por unidad de volumen de la aleación; crítica para aplicaciones aeroespaciales, automotrices y portátiles.",
        ),
        (
            "Módulo de Young (E)",
            f"{res['young']:.1f}",
            "GPa",
            "Rigidez elástica",
            "Pendiente de la zona elástica (esfuerzo/deformación); mide qué tan rígida es la aleación ante cargas reversibles.",
        ),
        (
            "Resistencia Específica (YS/ρ)",
            f"{res['ys_esp']:.1f}",
            "kN·m/kg",
            "Relación resistencia–peso",
            "Límite elástico dividido entre la densidad; cuantifica cuánta carga resiste el material por cada kilogramo.",
        ),
        (
            "Relación UTS / YS",
            f"{ratio_uts_ys:.2f}",
            "adim.",
            "Margen de endurecimiento",
            "Indica la capacidad de endurecimiento por deformación; valores mayores a 1.2 ofrecen mayor margen antes de fallar.",
        ),
    ]
    st.markdown(render_kpi_grid(kpis_mecanicos), unsafe_allow_html=True)

    st.markdown('<div class="sec-heading">🔬 Descriptores Físicos (Hume-Rothery)</div>', unsafe_allow_html=True)
    kpis_hr = [
        (
            "Electrones de Valencia (VEC)",
            f"{hr['HR_VEC']:.2f}",
            "e⁻/át.",
            "≥8 FCC · <6.87 BCC",
            "Promedio ponderado de electrones de valencia. Predice la estructura cristalina: ≥8 favorece FCC, <6.87 favorece BCC.",
        ),
        (
            "Distorsión Atómica (δ)",
            f"{hr['HR_delta']:.2f}",
            "%",
            "<6.6% Solución sólida",
            "Dispersión porcentual de radios atómicos. Valores bajos (<6.6%) favorecen soluciones sólidas; valores altos inducen fases intermetálicas o amorfas.",
        ),
        (
            "Temp. de Fusión (Tm)",
            f"{hr['HR_Tm'] - 273.15:.0f}",
            "°C",
            f"{hr['HR_Tm']:.0f} K · T/Tm = {res['t_hom']:.2f}",
            "Temperatura de fusión estimada por regla de mezclas y temperatura homóloga (T/Tm); encima de 0.5–0.6 predomina el ablandamiento y la termofluencia.",
        ),
        (
            "Entropía de Mezcla (ΔSmix)",
            f"{hr['HR_dSmix']:.2f}",
            "J/(mol·K)",
            ">12.5 Alta entropía",
            "Entropía configuracional ideal. Valores altos (>12.5 J/mol·K o ≥1.5R) estabilizan soluciones sólidas multicomponentes frente a compuestos ordenados.",
        ),
    ]
    st.markdown(render_kpi_grid(kpis_hr, extra_class="hr-accent"), unsafe_allow_html=True)

# --- GRÁFICAS INTERACTIVAS Y GUÍA EN PESTAÑAS ---
st.divider()
tab_ashby, tab_hr, tab_radar, tab_proc, tab_comp, tab_guia = st.tabs([
    "🗺️ Mapa de Ashby",
    "🔬 Mapa Hume-Rothery (VEC vs δ)",
    "🕸️ Comparativa Radar",
    "⚙️ Procesamiento y Temperatura",
    "📊 Composición (% at.)",
    "📖 Guía de Propiedades",
])

with tab_ashby:
    st.caption(
        "**Mapa de Ashby:** compara tu aleación (estrella amarilla) contra datos experimentales reales agrupados por fase "
        "para evaluar el compromiso clásico entre resistencia mecánica (YS) y ductilidad o densidad."
    )
    if df_ref is not None:
        tipo_mapa = st.radio(
            "Ejes del mapa de selección de materiales:",
            ["Resistencia (YS) vs. Ductilidad (Elongación %)", "Resistencia (YS) vs. Densidad (g/cm³)"],
            horizontal=True,
        )
        if "Ductilidad" in tipo_mapa:
            col_x, x_user, x_label = "elong", res["elong"], "Elongación / Ductilidad (%)"
        else:
            col_x, x_user, x_label = "dens", res["dens"], "Densidad (g/cm³)"

        df_plot = df_ref.dropna(subset=[col_x, "ys"])
        hover_cols = {"FORMULA": True, "phase": True}
        if "source" in df_plot.columns:
            hover_cols["source"] = True

        fig_ashby = px.scatter(
            df_plot,
            x=col_x,
            y="ys",
            color="phase",
            color_discrete_map=COLORES_FASE,
            hover_data=hover_cols,
            opacity=0.62,
            labels={col_x: x_label, "ys": "Límite Elástico YS (MPa)", "phase": "Estructura"},
            title=f"Tu aleación frente a {len(df_plot)} registros experimentales fusionados",
        )
        fig_ashby.add_trace(
            go.Scatter(
                x=[x_user],
                y=[res["ys"]],
                mode="markers+text",
                name="⭐ Tu Aleación",
                text=["Tu Aleación"],
                textposition="top center",
                textfont=dict(color="#fde047", size=12),
                marker=dict(symbol="star", size=20, color="#facc15", line=dict(width=2, color="#081224")),
            )
        )
        fig_ashby.update_layout(legend_title_text="Estructura / Referencia")
        st.plotly_chart(estilizar_plotly(fig_ashby, height=470), use_container_width=True)
    else:
        st.info("Sube `mpea_fusionado_ref.csv` a tu repositorio para habilitar el Mapa de Ashby actualizado.")

with tab_hr:
    st.caption(
        "**Mapa Hume-Rothery (Guo & Zhang):** ubica la aleación según su VEC (eje horizontal, que gobierna la red FCC/BCC) "
        "y su distorsión atómica δ (eje vertical, que separa soluciones sólidas simples de intermetálicos o vidrios metálicos)."
    )
    fig_hr = go.Figure()
    fig_hr.add_vrect(x0=2.0, x1=6.87, fillcolor="rgba(249, 115, 22, 0.13)", line_width=0, annotation_text="Zona BCC (VEC < 6.87)", annotation_position="top left")
    fig_hr.add_vrect(x0=6.87, x1=8.0, fillcolor="rgba(56, 189, 248, 0.14)", line_width=0, annotation_text="Dual FCC+BCC", annotation_position="top")
    fig_hr.add_vrect(x0=8.0, x1=11.5, fillcolor="rgba(37, 99, 235, 0.16)", line_width=0, annotation_text="Zona FCC (VEC ≥ 8.0)", annotation_position="top right")
    fig_hr.add_hline(y=6.6, line_dash="dash", line_color="#f472b6", annotation_text="Límite solución sólida (δ ≈ 6.6%): arriba predominan Intermetálicos / Amorfa", annotation_position="top left")

    ref_x, ref_y, ref_txt = [], [], []
    for pname, pdict in PRESETS.items():
        p_norm = {k: v / sum(pdict.values()) for k, v in pdict.items()}
        phr = calc_hr(p_norm)
        ref_x.append(phr["HR_VEC"])
        ref_y.append(phr["HR_delta"])
        ref_txt.append(pname.split("—")[0].strip())

    fig_hr.add_trace(
        go.Scatter(
            x=ref_x,
            y=ref_y,
            mode="markers+text",
            name="Aleaciones de referencia",
            text=ref_txt,
            textposition="bottom center",
            textfont=dict(color="#bfdbfe", size=11),
            marker=dict(size=10, color="#60a5fa", line=dict(width=1.2, color="#081224")),
        )
    )
    fig_hr.add_trace(
        go.Scatter(
            x=[hr["HR_VEC"]],
            y=[hr["HR_delta"]],
            mode="markers+text",
            name="⭐ Tu Aleación",
            text=[f"Tu Aleación ({res['fase']})"],
            textposition="top center",
            textfont=dict(color="#fde047", size=12),
            marker=dict(symbol="star", size=22, color="#facc15", line=dict(width=2, color="#081224")),
        )
    )
    fig_hr.update_layout(
        title="Mapa de Estabilidad de Fases (Criterio de Guo & Zhang: VEC vs. Distorsión δ)",
        xaxis_title="Concentración de Electrones de Valencia (VEC)",
        yaxis_title="Diferencia de Radio Atómico δ (%)",
        xaxis=dict(range=[2.5, 11.2]),
        yaxis=dict(range=[0, max(14.0, hr["HR_delta"] + 2.0)]),
    )
    st.plotly_chart(estilizar_plotly(fig_hr, height=460), use_container_width=True)

with tab_radar:
    st.caption(
        "**Comparativa Radar:** contrasta cinco métricas normalizadas (escala 0–100) frente a la aleación clásica de Cantor "
        "(CoCrFeMnNi) bajo el mismo método de manufactura y temperatura. Cuanto mayor sea el área cubierta, más equilibrado es el perfil."
    )
    cantor_comp = {e: 0.2 for e in ["Co", "Cr", "Fe", "Mn", "Ni"] if e in elementos_modelo}
    res_cantor = predecir_todo(cantor_comp, temperatura, proc_code)

    def puntuar_radar(r):
        return [
            min(r["ys"] / 1500.0 * 100.0, 100.0),
            min(r["uts"] / 1800.0 * 100.0, 100.0),
            min(r["hv"] / 700.0 * 100.0, 100.0),
            min(r["elong"] / 60.0 * 100.0, 100.0),
            min(max((14.0 - r["dens"]) / 10.0 * 100.0, 5.0), 100.0),
        ]

    ejes = ["Límite Elástico (YS)", "Resistencia Máx. (UTS)", "Dureza (HV)", "Ductilidad (%)", "Ligereza (1/ρ)"]
    vals_user = puntuar_radar(res)
    vals_cantor = puntuar_radar(res_cantor)

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=vals_cantor + [vals_cantor[0]],
            theta=ejes + [ejes[0]],
            fill="toself",
            name="Referencia: Cantor (CoCrFeMnNi)",
            line_color="#94a3b8",
            fillcolor="rgba(148, 163, 184, 0.22)",
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=vals_user + [vals_user[0]],
            theta=ejes + [ejes[0]],
            fill="toself",
            name="Tu Aleación",
            line_color="#38bdf8",
            fillcolor="rgba(56, 189, 248, 0.34)",
        )
    )
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(8, 18, 36, 0.72)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(56, 189, 248, 0.2)"),
            angularaxis=dict(gridcolor="rgba(56, 189, 248, 0.2)"),
        ),
        title=f"Perfil multidimensional vs. Aleación de Cantor ({proc_elegido_label}, {temperatura} °C)",
    )
    st.plotly_chart(estilizar_plotly(fig_radar, height=450), use_container_width=True)

with tab_proc:
    st.caption(
        "**Procesamiento y Temperatura:** la gráfica izquierda muestra cómo cambia el límite elástico (YS) según la ruta de "
        "manufactura elegida (barra celeste = proceso activo); la derecha muestra la caída de resistencia por ablandamiento térmico."
    )
    gp1, gp2 = st.columns(2)
    with gp1:
        ys_procs = [float(rf_ys.predict(construir_fila(comp_final, cols_ys, temp=temperatura, proc_code=p))[0]) for p in procesamientos]
        etiquetas_proc = [NOMBRES_PROC.get(p, p) for p in procesamientos]
        colores_bar = ["#38bdf8" if p == proc_code else "#1d4ed8" for p in procesamientos]
        fig_proc = go.Figure(
            data=[
                go.Bar(
                    x=etiquetas_proc,
                    y=ys_procs,
                    marker_color=colores_bar,
                    marker_line_color="#7dd3fc",
                    marker_line_width=1,
                    text=[f"{v:.0f}" for v in ys_procs],
                    textposition="outside",
                )
            ]
        )
        fig_proc.update_layout(
            title=f"Efecto del Procesamiento en YS a {temperatura} °C",
            xaxis_title="Método de procesamiento",
            yaxis_title="Límite Elástico YS (MPa)",
        )
        st.plotly_chart(estilizar_plotly(fig_proc, height=400), use_container_width=True)

    with gp2:
        temps_curva = [25, 200, 400, 600, 800, 1000, 1200]
        ys_curva = [float(rf_ys.predict(construir_fila(comp_final, cols_ys, temp=t, proc_code=proc_code))[0]) for t in temps_curva]
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(
                x=temps_curva,
                y=ys_curva,
                mode="lines+markers",
                name="Curva YS(T)",
                line=dict(color="#38bdf8", width=3),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.18)",
            )
        )
        fig_temp.add_trace(
            go.Scatter(
                x=[temperatura],
                y=[res["ys"]],
                mode="markers",
                name="Temperatura actual",
                marker=dict(size=14, color="#facc15", line=dict(width=2, color="#081224")),
            )
        )
        fig_temp.update_layout(
            title=f"Ablandamiento Térmico ({proc_elegido_label})",
            xaxis_title="Temperatura (°C)",
            yaxis_title="Límite Elástico YS (MPa)",
        )
        st.plotly_chart(estilizar_plotly(fig_temp, height=400), use_container_width=True)

with tab_comp:
    st.caption(
        "**Composición y Propiedades Elementales:** resume la fracción atómica (% at.) de cada elemento junto con sus constantes "
        "intrínsecas (VEC, radio atómico y punto de fusión) utilizadas para calcular los parámetros de Hume-Rothery."
    )
    df_comp = pd.DataFrame({
        "Elemento": list(comp_final.keys()),
        "Porcentaje Atómico (% at.)": [round(v * 100, 2) for v in comp_final.values()],
        "VEC": [ELEM_PROPS[k][0] for k in comp_final.keys()],
        "Radio Atómico (pm)": [ELEM_PROPS[k][1] for k in comp_final.keys()],
        "Tm (°C)": [round(ELEM_PROPS[k][3] - 273.15, 0) for k in comp_final.keys()],
    }).sort_values("Porcentaje Atómico (% at.)", ascending=False)

    cc1, cc2 = st.columns([1.1, 1.3])
    with cc1:
        paleta_azul = ["#38bdf8", "#2563eb", "#0ea5e9", "#60a5fa", "#1d4ed8", "#0284c7", "#93c5fd", "#1e40af"]
        fig_pie = px.pie(
            df_comp,
            names="Elemento",
            values="Porcentaje Atómico (% at.)",
            hole=0.48,
            color_discrete_sequence=paleta_azul,
            title="Distribución Atómica de la Aleación",
        )
        st.plotly_chart(estilizar_plotly(fig_pie, height=370), use_container_width=True)
    with cc2:
        st.markdown('<div class="sec-heading">Tabla de Composición y Propiedades Elementales</div>', unsafe_allow_html=True)
        st.dataframe(df_comp, hide_index=True, use_container_width=True)

with tab_guia:
    st.markdown('<div class="sec-heading">📋 Propiedades Mecánicas Estimadas</div>', unsafe_allow_html=True)
    tabla_mec = [
        (
            "Límite Elástico (YS)",
            "MPa",
            "Esfuerzo en el cual el material deja de comportarse elásticamente y comienza a deformarse de manera permanente.",
            "Criterio principal de diseño estructural para evitar deformaciones irreversibles en servicio.",
        ),
        (
            "Resistencia Máxima (UTS)",
            "MPa",
            "Esfuerzo máximo de tracción que soporta una probeta antes de iniciar la estricción o fracturarse.",
            "Define la carga última antes del colapso mecánico de la pieza.",
        ),
        (
            "Elongación / Ductilidad",
            "%",
            "Porcentaje de alargamiento plástico acumulado hasta el momento de la fractura en tracción.",
            "Valores altos indican capacidad para absorber energía y deformarse sin fractura frágil.",
        ),
        (
            "Dureza Vickers (HV)",
            "HV",
            "Resistencia superficial del material a ser penetrado por un indentador piramidal de diamante.",
            "Se correlaciona directamente con la resistencia al desgaste abrasivo y el límite elástico.",
        ),
        (
            "Densidad (ρ)",
            "g/cm³",
            "Masa por unidad de volumen calculada a partir de la composición atómica de la aleación.",
            "Determinante para aplicaciones donde el peso es crítico (aeroespacial, transporte, robótica).",
        ),
        (
            "Módulo de Young (E)",
            "GPa",
            "Relación entre esfuerzo y deformación dentro de la región elástica lineal.",
            "Mide la rigidez: a mayor E, menor deflexión elástica bajo la misma carga.",
        ),
        (
            "Resistencia Específica (YS/ρ)",
            "kN·m/kg",
            "Cociente entre el límite elástico y la densidad de la aleación.",
            "Permite comparar materiales ligeros frente a pesados en diseño de estructuras eficientes.",
        ),
        (
            "Relación UTS / YS",
            "adim.",
            "Cociente entre la resistencia máxima y el límite elástico.",
            "Mide el margen de endurecimiento por deformación; valores > 1.2 brindan mayor seguridad antes de la falla.",
        ),
    ]
    st.markdown(
        render_info_table(["Propiedad", "Unidad", "Significado físico", "Relevancia en ingeniería"], tabla_mec),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-heading">🔬 Descriptores Físicos de Hume-Rothery</div>', unsafe_allow_html=True)
    tabla_hr = [
        (
            "Electrones de Valencia (VEC)",
            "e⁻/át.",
            "Promedio ponderado de electrones de valencia (incluyendo electrones d) de los elementos aleantes.",
            "VEC ≥ 8.0 favorece fase FCC; VEC < 6.87 favorece BCC; entre 6.87 y 8.0 suelen coexistir FCC + BCC.",
        ),
        (
            "Distorsión Atómica (δ)",
            "%",
            "Desviación cuadrática media de los radios atómicos respecto al radio promedio de la aleación.",
            "δ < 6.6% favorece soluciones sólidas simples; valores mayores promueven intermetálicos o fases amorfas.",
        ),
        (
            "Temp. de Fusión (Tm) y T/Tm",
            "°C / adim.",
            "Punto de fusión estimado por regla de mezclas y temperatura homóloga (T en K dividida entre Tm en K).",
            "Estima el techo térmico de operación; arriba de T/Tm ≈ 0.5 se acelera la difusión y el ablandamiento.",
        ),
        (
            "Entropía de Mezcla (ΔSmix)",
            "J/(mol·K)",
            "Entropía configuracional ideal de la mezcla atómica aleatoria (-R Σ cᵢ ln cᵢ).",
            "Valores ≥ 12.5 J/(mol·K) (≈ 1.5R) caracterizan aleaciones de alta entropía que estabilizan soluciones sólidas.",
        ),
    ]
    st.markdown(
        render_info_table(["Descriptor", "Unidad", "Definición", "Criterio típico en MPEA"], tabla_hr),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-heading">🏭 Métodos de Procesamiento (Manufactura)</div>', unsafe_allow_html=True)
    tabla_proc = [
        (
            "Fundición",
            "Solidificación desde estado líquido (fusión por arco o inducción).",
            "Genera microestructuras dendríticas con granos relativamente gruesos; suele presentar segregación química moderada.",
        ),
        (
            "Recocido",
            "Tratamiento térmico controlado posterior a fundición o deformación.",
            "Homogeneiza la composición, alivia esfuerzos residuales y suele incrementar la ductilidad a costa de algo de YS.",
        ),
        (
            "Forjado / Laminado",
            "Deformación plástica en frío o en caliente (procesado termomecánico).",
            "Refina el tamaño de grano e introduce alta densidad de dislocaciones, elevando significativamente YS y UTS.",
        ),
        (
            "Pulvimetalurgia",
            "Aleado mecánico de polvos metálicos y consolidación por sinterizado (ej. SPS).",
            "Produce granos muy finos o nanocristalinos y excelente homogeneidad química, logrando altas resistencias.",
        ),
        (
            "Otro proceso",
            "Rutas no convencionales (manufactura aditiva, deposición láser, recubrimientos, etc.).",
            "Microestructuras altamente dependientes de la velocidad de enfriamiento y la historia térmica local.",
        ),
    ]
    st.markdown(
        render_info_table(["Proceso", "Descripción", "Efecto típico en propiedades"], tabla_proc),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-heading">📊 Descripción de las Gráficas y Tablas</div>', unsafe_allow_html=True)
    tabla_graf = [
        (
            "Celda Cristalina y Confianza",
            "Muestra la fase principal predicha (FCC, BCC, HCP, FCC + BCC, SS + Intermetálicos o Amorfa) junto con la probabilidad asignada por el modelo a las 6 clases.",
        ),
        (
            "Mapa de Ashby",
            "Ubica tu aleación sobre una base experimental real para visualizar compromisos entre resistencia (YS) frente a elongación (%) o densidad (g/cm³).",
        ),
        (
            "Mapa Hume-Rothery (VEC vs δ)",
            "Representa el criterio de estabilidad de fases de Guo & Zhang para verificar si la composición cae en zona FCC, BCC, dual o de alta distorsión.",
        ),
        (
            "Comparativa Radar",
            "Compara 5 ejes normalizados (YS, UTS, HV, ductilidad y ligereza) de tu diseño contra la aleación de referencia de Cantor (CoCrFeMnNi).",
        ),
        (
            "Procesamiento y Temperatura",
            "Evalúa la sensibilidad del límite elástico (YS) al cambiar de proceso de manufactura y al aumentar la temperatura de prueba de 25 °C a 1200 °C.",
        ),
        (
            "Tabla de Composición (% at.)",
            "Detalla la participación atómica de cada elemento y sus parámetros fundamentales (VEC, radio atómico en pm y temperatura de fusión en °C).",
        ),
    ]
    st.markdown(
        render_info_table(["Elemento visual / Tabla", "Qué muestra y cómo interpretarlo"], tabla_graf),
        unsafe_allow_html=True,
    )
