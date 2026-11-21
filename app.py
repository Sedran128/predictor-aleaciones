import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="Predictor de Aleaciones MPEA", page_icon="🧪", layout="centered")

@st.cache_resource
def cargar_modelos():
    return joblib.load('modelos_mpea.joblib')

datos = cargar_modelos()
rf_cls = datos['rf_cls']
rf_reg = datos['rf_reg']
elementos_modelo = sorted(datos['elementos'])
columnas_reg = datos['columnas_reg']
col_temp = datos['col_temp']

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

def construir_fila_reg(comp_dict, temp, proc_code):
    fila = {col: 0.0 for col in columnas_reg}
    for el, val in comp_dict.items():
        if el in fila:
            fila[el] = val
    fila[col_temp] = temp
    col_p = f'PROC_{proc_code}'
    if col_p in fila:
        fila[col_p] = 1.0
    return pd.DataFrame([fila])[columnas_reg]

# Encabezado
st.title("🧪 Predictor de Propiedades de Aleaciones (MPEA)")
st.markdown("Diseña tu aleación eligiendo composición, procesamiento y temperatura para estimar su **estructura cristalina** y **límite elástico (YS)**.")

# Selector de modo (reemplaza a las pestañas y elimina la casilla)
modo = st.radio(
    "Modo de entrada:",
    ["🎛️ Constructor Visual", "⌨️ Escribir Fórmula"],
    horizontal=True
)

comp_final = {}

if modo == "🎛️ Constructor Visual":
    preset = st.selectbox(
        "Cargar una aleación de ejemplo (opcional):",
        ["Personalizada (Al-Co-Cr-Fe-Ni)", "Aleación de Cantor (CoCrFeMnNi)", "Refractaria (NbTaTiVZr)", "Aluminio-Cobalto-Hierro-Níquel (AlCoFeNi)"]
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
        cols = st.columns(3)
        valores_crudos = {}
        for i, el in enumerate(seleccionados):
            with cols[i % 3]:
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

if st.button("🚀 Predecir Propiedades", type="primary", use_container_width=True):
    if not comp_final:
        st.error("⚠️ Ingresa al menos un elemento válido.")
    else:
        # 1. Predicción de fase cristalina
        fila_cls = {el: comp_final.get(el, 0.0) for el in datos['elementos']}
        X_cls_in = pd.DataFrame([fila_cls])
        fase_pred = rf_cls.predict(X_cls_in)[0]
        probs = rf_cls.predict_proba(X_cls_in)[0]
        clases = rf_cls.classes_
        
        # 2. Predicción de límite elástico (YS)
        X_reg_in = construir_fila_reg(comp_final, temperatura, proc_code)
        ys_pred = rf_reg.predict(X_reg_in)[0]
        
        # Resultados principales
        st.divider()
        res1, res2 = st.columns(2)
        res1.metric("Estructura Cristalina Predicha", fase_pred)
        res2.metric("Límite Elástico Estimado (YS)", f"{ys_pred:.1f} MPa")
        
        # Gráficas lado a lado
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Probabilidad de Fase (%)")
            df_probs = pd.DataFrame({'Fase': clases, 'Probabilidad (%)': probs * 100}).set_index('Fase')
            st.bar_chart(df_probs)
            
        with g2:
            st.subheader("Efecto del Procesamiento (MPa)")
            comparativa = {}
            for p in datos['procesamientos']:
                X_tmp = construir_fila_reg(comp_final, temperatura, p)
                comparativa[p] = rf_reg.predict(X_tmp)[0]
            df_proc_comp = pd.DataFrame({'Procesamiento': list(comparativa.keys()), 'YS (MPa)': list(comparativa.values())}).set_index('Procesamiento')
            st.bar_chart(df_proc_comp)
        
        with st.expander("Ver fracciones atómicas normalizadas (%)"):
            df_comp = pd.DataFrame({
                'Elemento': list(comp_final.keys()),
                'Fracción Atómica (%)': [round(v * 100, 2) for v in comp_final.values()]
            })
            st.dataframe(df_comp, hide_index=True, use_container_width=True)
