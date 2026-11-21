import os
import re
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Simulador MPEA | Materiales 2",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ESTILOS CSS ---
st.markdown(
    """
<style>
.crystal-card {
    background: linear-gradient(145deg, #161b22 0%, #1f2937 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    color: #f0f6fc;
}
.crystal-title {
    font-size: 1.08rem;
    font-weight: 700;
    margin-top: 6px;
    margin-bottom: 2px;
    color: #f0f6fc;
}
.crystal-sub {
    font-size: 0.82rem;
    color: #9ca3af;
    margin-bottom: 0;
}
div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 10px;
    padding: 10px 14px;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- CARGA DE MODELOS Y DATASET DE REFERENCIA ---
@st.cache_resource
def cargar_modelos():
    return joblib.load("modelos_mpea.joblib")


@st.cache_data
def cargar_dataset_referencia():
    if not os.path.exists("MPEA_dataset.csv"):
        return None
    df = pd.read_csv("MPEA_dataset.csv")
    col_ys = "PROPERTY: YS (MPa)"
    col_el = "PROPERTY: Elongation (%)"
    col_den = r"PROPERTY: Calculated Density (g/cm$^3$)"
    col_ph = "PROPERTY: BCC/FCC/other"
    for c in [col_ys, col_el, col_den]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    cols_keep = [c for c in ["FORMULA", col_ph, col_ys, col_el, col_den] if c in df.columns]
    return df[cols_keep].dropna(subset=[col_ys, col_ph])


datos = cargar_modelos()
df_ref = cargar_dataset_referencia()

rf_cls = datos["rf_cls"]
rf_ys = datos["rf_ys"]
rf_uts = datos["rf_uts"]
rf_elong = datos["rf_elong"]
rf_hv = datos["rf_hv"]
rf_dens = datos["rf_dens"]
rf_young = datos["rf_young"]

elementos_modelo = sorted(datos["elementos"])
col_temp = datos["col_temp"]
cols_ys = datos["cols_ys"]
cols_hv = datos["cols_hv"]

NOMBRES_PROC = {
    "CAST": "Fundición (CAST)",
    "ANNEAL": "Recocido (ANNEAL)",
    "WROUGHT": "Forjado / Laminado (WROUGHT)",
    "POWDER": "Pulvimetalurgia (POWDER)",
    "OTHER": "Otro procesamiento (OTHER)",
}
INV_PROC = {v: k for k, v in NOMBRES_PROC.items()}

PRESETS = {
    "Aleación de Cantor (CoCrFeMnNi)": {"Co": 20.0, "Cr": 20.0, "Fe": 20.0, "Mn": 20.0, "Ni": 20.0},
    "AlCoCrFeNi (Equiatómica)": {"Al": 20.0, "Co": 20.0, "Cr": 20.0, "Fe": 20.0, "Ni": 20.0},
    "AlCoFeNi (Alta Resistencia BCC)": {"Al": 25.0, "Co": 25.0, "Fe": 25.0, "Ni": 25.0},
    "Refractaria (NbTaTiVZr)": {"Nb": 20.0, "Ta": 20.0, "Ti": 20.0, "V": 20.0, "Zr": 20.0},
    "Ligera rica en Al (Al35Cr15Fe15Ni35)": {"Al": 35.0, "Cr": 15.0, "Fe": 15.0, "Ni": 35.0},
}


# --- FUNCIONES DE PREDICCIÓN ---
def parse_formula(formula):
    matches = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    comp = {}
    for el, qty in matches:
        val = float(qty) if qty != "" else 1.0
        comp[el] = comp.get(el, 0.0) + val
    total = sum(comp.values())
    if total > 0:
        comp = {k: v / total for k, v in comp.items()}
    return comp


def construir_df_comp(comp_dict):
    fila = {el: comp_dict.get(el, 0.0) for el in datos["elementos"]}
    return pd.DataFrame([fila])[datos["elementos"]]


def construir_df_proc(comp_dict, proc_code):
    fila = {col: 0.0 for col in cols_hv}
    for el, val in comp_dict.items():
        if el in fila:
            fila[el] = val
    col_p = f"PROC_{proc_code}"
    if col_p in fila:
        fila[col_p] = 1.0
    return pd.DataFrame([fila])[cols_hv]


def construir_df_temp_proc(comp_dict, temp, proc_code):
    fila = {col: 0.0 for col in cols_ys}
    for el, val in comp_dict.items():
        if el in fila:
            fila[el] = val
    fila[col_temp] = temp
    col_p = f"PROC_{proc_code}"
    if col_p in fila:
        fila[col_p] = 1.0
    return pd.DataFrame([fila])[cols_ys]


def predecir_todo(comp_dict, temp, proc_code):
    X_comp = construir_df_comp(comp_dict)
    X_proc = construir_df_proc(comp_dict, proc_code)
    X_tp = construir_df_temp_proc(comp_dict, temp, proc_code)

    fase = rf_cls.predict(X_comp)[0]
    probs = dict(zip(rf_cls.classes_, rf_cls.predict_proba(X_comp)[0]))
    ys = float(rf_ys.predict(X_tp)[0])
    uts = float(rf_uts.predict(X_tp)[0])
    elong = float(rf_elong.predict(X_tp)[0])
    hv = float(rf_hv.predict(X_proc)[0])
    dens = float(rf_dens.predict(X_comp)[0])
    young = float(rf_young.predict(X_comp)[0])
    ys_esp = ys / dens if dens > 0 else 0.0
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
    }


def svg_estructura_cristalina(fase):
    if fase == "FCC":
        svg = """
<svg viewBox="0 0 200 200" width="160" height="160" xmlns="http://www.w3.org/2000/svg">
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
<g stroke="#38bdf8" stroke-width="1" stroke-dasharray="3,3" opacity="0.55">
<line x1="35" y1="75" x2="125" y2="165"/>
<line x1="125" y1="75" x2="35" y2="165"/>
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
        sub = "Cúbica Centrada en las Caras · Alta ductilidad"
    elif fase == "BCC":
        svg = """
<svg viewBox="0 0 200 200" width="160" height="160" xmlns="http://www.w3.org/2000/svg">
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
        sub = "Cúbica Centrada en el Cuerpo · Alta resistencia y dureza"
    else:
        svg = """
<svg viewBox="0 0 200 200" width="160" height="160" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="gDual1" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#3b82f6" stop-opacity="0.35"/>
<stop offset="100%" stop-color="#06b6d4" stop-opacity="0.15"/>
</linearGradient>
<linearGradient id="gDual2" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#f97316" stop-opacity="0.4"/>
<stop offset="100%" stop-color="#a855f7" stop-opacity="0.2"/>
</linearGradient>
</defs>
<circle cx="100" cy="100" r="82" fill="#0f172a" stroke="#a855f7" stroke-width="2.5"/>
<path d="M 32 100 Q 65 40 120 42 Q 105 95 95 110 Q 55 125 32 100 Z" fill="url(#gDual1)" stroke="#60a5fa" stroke-width="1.5"/>
<path d="M 120 42 Q 170 65 168 120 Q 125 135 95 110 Q 105 95 120 42 Z" fill="url(#gDual2)" stroke="#fb923c" stroke-width="1.5"/>
<path d="M 32 100 Q 55 125 95 110 Q 125 135 168 120 Q 130 175 65 165 Q 35 140 32 100 Z" fill="#a855f7" fill-opacity="0.22" stroke="#c084fc" stroke-width="1.5"/>
<circle cx="72" cy="80" r="6" fill="#67e8f9"/>
<circle cx="132" cy="85" r="7" fill="#fdba74"/>
<circle cx="115" cy="142" r="6.5" fill="#e879f9"/>
<circle cx="75" cy="138" r="5" fill="#93c5fd"/>
<circle cx="145" cy="112" r="4.5" fill="#fef08a"/>
<text x="58" y="68" fill="#bfdbfe" font-size="11" font-family="sans-serif" font-weight="bold">FCC</text>
<text x="122" y="70" fill="#fed7aa" font-size="11" font-family="sans-serif" font-weight="bold">BCC</text>
<text x="72" y="160" fill="#e9d5ff" font-size="10" font-family="sans-serif" font-weight="bold">Intermetálicos</text>
</svg>
"""
        titulo = "Multifase / Compleja (other)"
        sub = "Coexistencia FCC + BCC, HCP o precipitados intermetálicos"

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
    st.session_state["saved_preset"] = "Aleación de Cantor (CoCrFeMnNi)"
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


# --- BARRA LATERAL DE CONFIGURACIÓN ---
with st.sidebar:
    st.header("⚙️ Diseño de Aleación")
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
            "Aleación base (preset):",
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
                if abs(suma_pct - 100.0) <= 0.2:
                    st.success(f"Suma total: **{suma_pct:.1f}% at.**")
                else:
                    st.warning(f"Suma actual: **{suma_pct:.1f}%** (se normalizará automáticamente a 100%)")
                comp_final = {k: v / suma_pct for k, v in valores_pct.items() if v > 0}
                formula_mostrar = " ".join([f"{k}{round(v * 100, 1)}" for k, v in comp_final.items()])
        else:
            st.warning("Selecciona al menos un elemento.")
    else:
        formula_texto = st.text_input(
            "Fórmula química (proporciones molares o %):",
            value="Al0.2 Co1 Cr1 Fe1 Mn1 Ni1",
            help="Ejemplo: Al1 Co1 Fe1 Ni1 o Co20 Cr20 Fe20 Mn20 Ni20",
        )
        comp_final = parse_formula(formula_texto)
        formula_mostrar = " ".join([f"{k}{round(v * 100, 1)}" for k, v in comp_final.items()])

    st.divider()
    opciones_proc = [NOMBRES_PROC.get(p, p) for p in datos["procesamientos"]]
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
st.title("🧪 Simulador Predictivo de Aleaciones de Alta Entropía (MPEA)")

if not comp_final:
    st.info("👈 Selecciona elementos en el panel izquierdo para calcular las propiedades de tu aleación.")
    st.stop()

st.caption(
    f"**Composición normalizada (% at.):** `{formula_mostrar}` · "
    f"**Procesamiento:** {proc_elegido_label} · **Temperatura:** {temperatura} °C"
)

# Predicción instantánea en tiempo real
res = predecir_todo(comp_final, temperatura, proc_code)

# --- BLOQUE SUPERIOR: CELDA CRISTALINA SVG + FICHA TÉCNICA ---
col_celda, col_ficha = st.columns([1.15, 2.4], gap="large")

with col_celda:
    st.markdown(svg_estructura_cristalina(res["fase"]), unsafe_allow_html=True)
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    st.caption("**Probabilidad estimada por fase:**")
    for ph in ["FCC", "BCC", "other"]:
        p_val = float(res["probs"].get(ph, 0.0))
        etiqueta = "Multifase (other)" if ph == "other" else ph
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:2px;'>"
            f"<span>{etiqueta}</span><strong>{p_val*100:.1f}%</strong></div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(p_val, 0.0), 1.0))

with col_ficha:
    st.subheader("📋 Ficha Técnica Estimada")
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Límite Elástico (YS)", f"{res['ys']:.1f} MPa")
    r1c2.metric("Resistencia Máxima (UTS)", f"{res['uts']:.1f} MPa")
    r1c3.metric("Elongación / Ductilidad", f"{res['elong']:.1f} %")

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric(
        "Dureza Vickers (HV)",
        f"{res['hv']:.0f} HV",
        help="Estimada a temperatura ambiente según composición y procesamiento.",
    )
    r2c2.metric("Densidad (ρ)", f"{res['dens']:.2f} g/cm³")
    r2c3.metric("Módulo de Young (E)", f"{res['young']:.1f} GPa")

    r3c1, r3c2 = st.columns(2)
    r3c1.metric(
        "Resistencia Específica (YS / ρ)",
        f"{res['ys_esp']:.1f} kN·m/kg",
        help="Relación resistencia-peso: clave para aplicaciones ligeras y aeroespaciales.",
    )
    ratio_uts_ys = res["uts"] / res["ys"] if res["ys"] > 0 else 1.0
    r3c2.metric(
        "Relación UTS / YS",
        f"{ratio_uts_ys:.2f}",
        help="Valores mayores a 1 indican mayor margen de endurecimiento por deformación antes de la fractura.",
    )

# --- GRÁFICAS INTERACTIVAS EN PESTAÑAS ---
st.divider()
tab_ashby, tab_radar, tab_proc, tab_comp = st.tabs([
    "🗺️ Mapa de Ashby",
    "🕸️ Comparativa Radar",
    "⚙️ Procesamiento y Temperatura",
    "📊 Composición (% at.)",
])

with tab_ashby:
    if df_ref is not None:
        tipo_mapa = st.radio(
            "Ejes del mapa de selección de materiales:",
            ["Resistencia (YS) vs. Ductilidad (Elongación %)", "Resistencia (YS) vs. Densidad (g/cm³)"],
            horizontal=True,
        )
        col_ys_csv = "PROPERTY: YS (MPa)"
        col_ph_csv = "PROPERTY: BCC/FCC/other"
        colores_fase = {"FCC": "#3b82f6", "BCC": "#f97316", "other": "#a855f7"}

        if "Ductilidad" in tipo_mapa:
            col_x = "PROPERTY: Elongation (%)"
            df_plot = df_ref.dropna(subset=[col_x, col_ys_csv])
            x_user, x_label = res["elong"], "Elongación / Ductilidad (%)"
        else:
            col_x = r"PROPERTY: Calculated Density (g/cm$^3$)"
            df_plot = df_ref.dropna(subset=[col_x, col_ys_csv])
            x_user, x_label = res["dens"], "Densidad (g/cm³)"

        fig_ashby = px.scatter(
            df_plot,
            x=col_x,
            y=col_ys_csv,
            color=col_ph_csv,
            color_discrete_map=colores_fase,
            hover_data={"FORMULA": True, col_ph_csv: True},
            opacity=0.55,
            labels={col_x: x_label, col_ys_csv: "Límite Elástico YS (MPa)", col_ph_csv: "Fase"},
            title="Posición de tu aleación frente a los datos experimentales MPEA",
        )
        fig_ashby.add_trace(
            go.Scatter(
                x=[x_user],
                y=[res["ys"]],
                mode="markers+text",
                name="⭐ Tu Aleación",
                text=["Tu Aleación"],
                textposition="top center",
                marker=dict(symbol="star", size=20, color="#facc15", line=dict(width=2, color="#111827")),
            )
        )
        fig_ashby.update_layout(height=460, legend_title_text="Fase / Referencia")
        st.plotly_chart(fig_ashby, use_container_width=True)
    else:
        st.info("Sube `MPEA_dataset.csv` a tu repositorio para habilitar el Mapa de Ashby interactivo.")

with tab_radar:
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
            fillcolor="rgba(148, 163, 184, 0.2)",
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=vals_user + [vals_user[0]],
            theta=ejes + [ejes[0]],
            fill="toself",
            name="Tu Aleación",
            line_color="#38bdf8",
            fillcolor="rgba(56, 189, 248, 0.3)",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Perfil multidimensional vs. Aleación de Cantor ({proc_elegido_label}, {temperatura} °C)",
        height=450,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab_proc:
    gp1, gp2 = st.columns(2)
    with gp1:
        procs = datos["procesamientos"]
        ys_procs = [float(rf_ys.predict(construir_df_temp_proc(comp_final, temperatura, p))[0]) for p in procs]
        colores_bar = ["#facc15" if p == proc_code else "#3b82f6" for p in procs]
        fig_proc = go.Figure(
            data=[
                go.Bar(
                    x=procs,
                    y=ys_procs,
                    marker_color=colores_bar,
                    text=[f"{v:.0f}" for v in ys_procs],
                    textposition="outside",
                )
            ]
        )
        fig_proc.update_layout(
            title=f"Efecto del Procesamiento en YS a {temperatura} °C",
            xaxis_title="Método de procesamiento",
            yaxis_title="Límite Elástico YS (MPa)",
            height=400,
        )
        st.plotly_chart(fig_proc, use_container_width=True)

    with gp2:
        temps_curva = [25, 200, 400, 600, 800, 1000, 1200]
        ys_curva = [float(rf_ys.predict(construir_df_temp_proc(comp_final, t, proc_code))[0]) for t in temps_curva]
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(
                x=temps_curva,
                y=ys_curva,
                mode="lines+markers",
                name="Curva YS(T)",
                line=dict(color="#f97316", width=3),
                fill="tozeroy",
                fillcolor="rgba(249, 115, 22, 0.15)",
            )
        )
        fig_temp.add_trace(
            go.Scatter(
                x=[temperatura],
                y=[res["ys"]],
                mode="markers",
                name="Temperatura actual",
                marker=dict(size=14, color="#facc15", line=dict(width=2, color="#111827")),
            )
        )
        fig_temp.update_layout(
            title=f"Ablandamiento Térmico ({proc_elegido_label})",
            xaxis_title="Temperatura (°C)",
            yaxis_title="Límite Elástico YS (MPa)",
            height=400,
        )
        st.plotly_chart(fig_temp, use_container_width=True)

with tab_comp:
    df_comp = pd.DataFrame({
        "Elemento": list(comp_final.keys()),
        "Porcentaje Atómico (% at.)": [round(v * 100, 2) for v in comp_final.values()],
    }).sort_values("Porcentaje Atómico (% at.)", ascending=False)

    cc1, cc2 = st.columns([1.2, 1])
    with cc1:
        fig_pie = px.pie(
            df_comp,
            names="Elemento",
            values="Porcentaje Atómico (% at.)",
            hole=0.45,
            title="Distribución Atómica de la Aleación",
        )
        fig_pie.update_layout(height=360)
        st.plotly_chart(fig_pie, use_container_width=True)
    with cc2:
        st.markdown("#### Tabla de Composición")
        st.dataframe(df_comp, hide_index=True, use_container_width=True)
