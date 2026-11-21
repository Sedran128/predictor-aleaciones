import streamlit as st
import pandas as pd
import joblib
import re

# Configuración de la página
st.set_page_config(page_title="Predictor de Aleaciones MPEA", page_icon="🧪", layout="centered")

@st.cache_resource
def cargar_modelos():
    return joblib.load('modelos_mpea.joblib')

datos = cargar_modelos()
rf_cls = datos['rf_cls']
rf_reg = datos['rf_reg']
elementos_modelo = sorted(datos['elementos'])

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

# Encabezado
st.title("🧪 Predictor de Propiedades de Aleaciones (MPEA)")
st.markdown("Diseña tu aleación seleccionando los elementos y la temperatura para estimar su **estructura cristalina** y **límite elástico (YS)**.")

# Pestañas para elegir el modo de entrada
tab1, tab2 = st.tabs(["🎛️ Constructor Visual", "⌨️ Escribir Fórmula"])

comp = {}

with tab1:
    # Botones de aleaciones predefinidas para demostración rápida
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

    # Selector múltiple de elementos
    seleccionados = st.multiselect(
        "1. Selecciona los elementos químicos de tu aleación:",
        options=elementos_modelo,
        default=default_els
    )

    # Sliders dinámicos en columnas para cada elemento elegido
    if seleccionados:
        st.write("2. Ajusta la proporción molar de cada elemento:")
        cols = st.columns(3)
        valores_crudos = {}
        for i, el in enumerate(seleccionados):
            with cols[i % 3]:
                valores_crudos[el] = st.slider(
                    f"{el}", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key=f"slider_{el}"
                )
        
        # Normalizar automáticamente a fracciones atómicas
        total = sum(valores_crudos.values())
        comp_visual = {k: v / total for k, v in valores_crudos.items()}
        
        # Mostrar la fórmula resultante en tiempo real
        formula_generada = " ".join([f"{k}{round(v, 2)}" for k, v in valores_crudos.items()])
        st.info(f"**Fórmula generada:** `{formula_generada}`")
        comp = comp_visual
    else:
        st.warning("Selecciona al menos un elemento para comenzar.")

with tab2:
    formula_texto = st.text_input(
        "Escribe la fórmula química directamente:",
        value="Al0.5 Co1 Cr1 Fe1 Ni1",
        help="Ejemplo: Co1 Cr1 Fe1 Mn1 Ni1"
    )
    comp_texto = parse_formula(formula_texto)

# Selector de temperatura (común para ambas pestañas)
st.divider()
temperatura = st.slider("🌡️ Temperatura de prueba (°C)", min_value=25, max_value=1200, value=25, step=25)

# Selector de qué pestaña usar al predecir
modo_texto = st.checkbox("Usar la fórmula escrita en la pestaña 'Escribir Fórmula' en lugar de los sliders", value=False)
comp_final = comp_texto if modo_texto else comp

# Botón de predicción
if st.button("🚀 Predecir Propiedades", type="primary", use_container_width=True):
    if not comp_final:
        st.error("⚠️ Ingresa al menos un elemento válido.")
    else:
        # Alinear columnas con el modelo
        fila = {el: comp_final.get(el, 0.0) for el in datos['elementos']}
        X_cls_in = pd.DataFrame([fila])
        
        # 1. Predicción de fase
        fase_pred = rf_cls.predict(X_cls_in)[0]
        probs = rf_cls.predict_proba(X_cls_in)[0]
        clases = rf_cls.classes_
        
        # 2. Predicción de límite elástico (YS)
        X_reg_in = X_cls_in.copy()
        X_reg_in['PROPERTY: Test temperature ($^\\circ$C)'] = temperatura
        ys_pred = rf_reg.predict(X_reg_in)[0]
        
        # Mostrar resultados principales
        st.divider()
        res1, res2 = st.columns(2)
        res1.metric("Estructura Cristalina Predicha", fase_pred)
        res2.metric("Límite Elástico Estimado (YS)", f"{ys_pred:.1f} MPa")
        
        # Gráfica de probabilidades por fase
        st.subheader("Probabilidad por fase cristalina")
        df_probs = pd.DataFrame({'Fase': clases, 'Probabilidad (%)': probs * 100}).set_index('Fase')
        st.bar_chart(df_probs)
        
        # Desglose de composición normalizada
        with st.expander("Ver fracciones atómicas normalizadas (%)"):
            df_comp = pd.DataFrame({
                'Elemento': list(comp_final.keys()),
                'Fracción Atómica (%)': [round(v * 100, 2) for v in comp_final.values()]
            })
            st.dataframe(df_comp, hide_index=True, use_container_width=True)
