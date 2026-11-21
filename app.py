import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="Predictor de Aleaciones MPEA", page_icon="🧪", layout="wide")

@st.cache_resource
def cargar_modelos():
    return joblib.load('modelos_mpea.joblib')

datos = cargar_modelos()
rf_cls   = datos['rf_cls']
rf_ys    = datos['rf_ys']
rf_uts   = datos['rf_uts']
rf_elong = datos['rf_elong']
rf_hv    = datos['rf_hv']
rf_dens  = datos['rf_dens']
rf_young = datos['rf_young']

elementos_modelo = sorted(datos['elementos'])
col_temp = datos['col_temp']
cols_ys  = datos['cols_ys']
cols_hv  = datos['cols_hv']

NOMBRES_PROC = {
    'CAST': 'Fundición (CAST)',
    'ANNEAL': 'Recocido (ANNEAL)',
    'WROUGHT': 'Forjado / Laminado (WROUGHT)',
    'POWDER': 'Pulvimetalurgia (POWDER)',
    'OTHER': 'Otro procesamiento (OTHER)'
}
INV_PROC = {v: k for k, v in NOMBRES_PROC.items()}

def parse_formula(formula):
    matches = re.findall(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)', formula)
    comp = {}
    for el, qty in matches:
        val = float(qty) if qty != '' else 1.0
        comp[el] = comp.get(el, 0.0) + val
    total = sum(comp.values())
    if total > 0:
        comp = {k: v / total for k, v in comp.items()}
    return comp

def construir_df_comp(comp_dict):
    fila = {el: comp_dict.get(el, 0.0) for el in datos['elementos']}
    return pd.DataFrame([fila])[datos['elementos']]

def construir_df_proc(comp_dict, proc_code):
    fila = {col: 0.0 for col in cols_hv}
    for el, val in comp_dict.items():
        if el in fila:
            fila[el] = val
    col_p = f'PROC_{proc_code}'
    if col_p in fila:
        fila[col_p] = 1.0
    return pd.DataFrame([fila])[cols_hv]

def construir_df_temp_proc(comp_dict, temp, proc_code):
    fila = {col: 0.0 for col in cols_ys}
    for el, val in comp_dict.items():
        if el in fila:
            fila[el] = val
    fila[col_temp] = temp
    col_p = f'PROC_{proc_code}'
    if col_p in fila:
        fila[col_p] = 1.0
    return pd.DataFrame([fila])[cols_ys]

# Encabezado
st.title("🧪 Simulador Predictivo de Aleaciones de Alta Entropía (MPEA)")
st.markdown("Configura la composición, el método de procesamiento y la temperatura para estimar la **ficha técnica completa** de tu aleación.")

# Selector de modo sin casillas extra
modo = st.radio(
    "Modo de entrada:",
    ["🎛️ Constructor Visual", "⌨️ Escribir Fórmula"],
    horizontal=True
)

comp_final = {}

if modo == "🎛️ Constructor Visual":
    preset = st.selectbox(
        "Cargar una aleación de ejemplo (opcional):",
        [
            "Personalizada (Al-Co-Cr-Fe-Ni)",
            "Aleación de Cantor (CoCrFeMnNi)",
            "Refractaria (NbTaTiVZr)",
            "Aluminio-Cobalto-Hierro-Níquel (AlCoFeNi)"
        ]
    )
    
    if preset == "Aleación de Cantor (CoCrFeMnNi)":
        default_els = [e for e in ['Co', 'Cr', 'Fe', 'Mn', 'Ni'] if e in elementos_modelo]
    elif preset == "Refractaria (NbTaTiVZr)":
        default_els = [e for e in ['Nb', 'Ta', 'Ti', 'V', 'Zr'] if e in elementos_modelo]
    elif preset == "Aluminio-Cobalto-Hierro-Níquel (AlCoFeNi)":
        default_els = [e for e in ['Al', 'Co', 'Fe', 'Ni'] if e in elementos_modelo]
    else:
        default_els = [e for e in ['Al', 'Co', 'Cr', 'Fe', 'Ni'] if e in elementos_modelo]

    seleccionados = st.multiselect(
        "1. Selecciona los elementos químicos de tu aleación:",
        options=elementos_modelo,
        default=default_els
    )

    if seleccionados:
        st.write("2. Ajusta la proporción molar de cada elemento:")
        cols = st.columns(4)
        valores_crudos = {}
        for i, el in enumerate(seleccionados):
            with cols[i % 4]:
                valores_crudos[el] = st.slider(
                    f"{el}", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key=f"slider_{el}"
                )
        
        total = sum(valores_crudos.values())
        comp_final = {k: v / total for k, v in valores_crudos.items()}
        formula_generada = " ".join([f"{k}{round(v, 2)}" for k, v in valores_crudos.items()])
        st.info(f"**Fórmula generada:** `{formula_generada}`")
    else:
        st.warning("Selecciona al menos un elemento para comenzar.")
else:
    formula_texto = st.text_input(
        "Escribe la fórmula química directamente:",
        value="Al0.2 Co1 Cr1 Fe1 Mn1 Ni1",
        help="Ejemplo: Al1 Co1 Fe1 Ni1"
    )
    comp_final = parse_formula(formula_texto)

# Condiciones de procesamiento y prueba
st.divider()
c1, c2 = st.columns(2)
with c1:
    opciones_proc = [NOMBRES_PROC.get(p, p) for p in datos['procesamientos']]
    proc_elegido_label = st.selectbox("⚙️ Método de procesamiento:", opciones_proc)
    proc_code = INV_PROC.get(proc_elegido_label, proc_elegido_label)
with c2:
    temperatura = st.slider("🌡️ Temperatura de prueba (°C)", min_value=25, max_value=1200, value=25, step=25)

if st.button("🚀 Predecir Ficha Técnica Completa", type="primary", use_container_width=True):
    if not comp_final:
        st.error("⚠️ Ingresa al menos un elemento válido.")
    else:
        # Construir entradas para cada tipo de modelo
        X_comp      = construir_df_comp(comp_final)
        X_proc      = construir_df_proc(comp_final, proc_code)
        X_temp_proc = construir_df_temp_proc(comp_final, temperatura, proc_code)

        # Ejecutar las 7 predicciones
        fase_pred  = rf_cls.predict(X_comp)[0]
        probs      = rf_cls.predict_proba(X_comp)[0]
        clases     = rf_cls.classes_

        ys_pred    = rf_ys.predict(X_temp_proc)[0]
        uts_pred   = rf_uts.predict(X_temp_proc)[0]
        elong_pred = rf_elong.predict(X_temp_proc)[0]
        hv_pred    = rf_hv.predict(X_proc)[0]
        dens_pred  = rf_dens.predict(X_comp)[0]
        young_pred = rf_young.predict(X_comp)[0]
        ys_esp     = ys_pred / dens_pred if dens_pred > 0 else 0.0

        # Ficha técnica: Fila 1 (Propiedades Mecánicas)
        st.divider()
        st.subheader("📋 Ficha Técnica Estimada")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estructura Cristalina", fase_pred)
        m2.metric("Límite Elástico (YS)", f"{ys_pred:.1f} MPa")
        m3.metric("Resistencia Máxima (UTS)", f"{uts_pred:.1f} MPa")
        m4.metric("Elongación / Ductilidad", f"{elong_pred:.1f} %")

        # Ficha técnica: Fila 2 (Propiedades Físicas y Desempeño)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Dureza Vickers (HV)", f"{hv_pred:.0f} HV", help="Estimada a temperatura ambiente según composición y procesamiento.")
        p2.metric("Densidad (ρ)", f"{dens_pred:.2f} g/cm³")
        p3.metric("Módulo de Young (E)", f"{young_pred:.1f} GPa")
        p4.metric("Resistencia Específica (YS/ρ)", f"{ys_esp:.1f} kN·m/kg", help="Relación resistencia-peso: clave para aplicaciones ligeras.")

        # Gráficas de análisis
        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Probabilidad por Fase Cristalina (%)")
            df_probs = pd.DataFrame({'Fase': clases, 'Probabilidad (%)': probs * 100}).set_index('Fase')
            st.bar_chart(df_probs)
            
        with g2:
            st.subheader(f"Efecto del Procesamiento en YS a {temperatura} °C (MPa)")
            comparativa = {}
            for p in datos['procesamientos']:
                X_tmp = construir_df_temp_proc(comp_final, temperatura, p)
                comparativa[p] = rf_ys.predict(X_tmp)[0]
            df_proc_comp = pd.DataFrame({'Procesamiento': list(comparativa.keys()), 'YS (MPa)': list(comparativa.values())}).set_index('Procesamiento')
            st.bar_chart(df_proc_comp)

        # Curva de comportamiento térmico
        st.subheader(f"Curva de Ablandamiento Térmico ({proc_elegido_label})")
        temps_curva = [25, 200, 400, 600, 800, 1000, 1200]
        ys_curva = [rf_ys.predict(construir_df_temp_proc(comp_final, t, proc_code))[0] for t in temps_curva]
        df_curva = pd.DataFrame({'Temperatura (°C)': temps_curva, 'Límite Elástico (MPa)': ys_curva}).set_index('Temperatura (°C)')
        st.line_chart(df_curva)

        with st.expander("Ver fracciones atómicas normalizadas (%)"):
            df_comp = pd.DataFrame({
                'Elemento': list(comp_final.keys()),
                'Fracción Atómica (%)': [round(v * 100, 2) for v in comp_final.values()]
            })
            st.dataframe(df_comp, hide_index=True, use_container_width=True)
