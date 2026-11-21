import streamlit as st
import pandas as pd
import joblib
import re

# Configuración de la página
st.set_page_config(page_title="Predictor de Aleaciones MPEA", page_icon="🧪", layout="centered")

# Cargar modelos una sola vez
@st.cache_resource
def cargar_modelos():
    return joblib.load('modelos_mpea.joblib')

datos = cargar_modelos()
rf_cls = datos['rf_cls']
rf_reg = datos['rf_reg']
elementos_modelo = datos['elementos']

# Función para leer la fórmula química
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

# Interfaz visual
st.title("🧪 Predictor de Propiedades de Aleaciones (MPEA)")
st.write("Ingresa la composición química y la temperatura de prueba para estimar la **estructura cristalina** y el **límite elástico (YS)**.")

# Entradas del usuario
col1, col2 = st.columns([2, 1])
with col1:
    formula = st.text_input("Fórmula química", value="Al0.5 Co1 Cr1 Fe1 Ni1", help="Ejemplo: Co1 Cr1 Fe1 Mn1 Ni1")
with col2:
    temperatura = st.slider("Temperatura (°C)", min_value=25, max_value=1200, value=25, step=25)

# Ejemplos rápidos
st.caption("Ejemplos que puedes probar: `Co1 Cr1 Fe1 Mn1 Ni1` | `Nb1 Ta1 Ti1 V1 Zr1` | `Al1 Co1 Fe1 Ni1`")

if st.button("Predecir propiedades", type="primary"):
    comp = parse_formula(formula)
    
    if not comp:
        st.error("⚠️ Fórmula no válida. Usa símbolos químicos como: Al0.5 Co1 Cr1 Fe1 Ni1")
    else:
        # Alinear columnas con el entrenamiento
        fila = {el: comp.get(el, 0.0) for el in elementos_modelo}
        X_cls_in = pd.DataFrame([fila])
        
        # 1. Predicción de fase
        fase_pred = rf_cls.predict(X_cls_in)[0]
        probs = rf_cls.predict_proba(X_cls_in)[0]
        clases = rf_cls.classes_
        
        # 2. Predicción de límite elástico (YS)
        X_reg_in = X_cls_in.copy()
        X_reg_in['PROPERTY: Test temperature ($^\\circ$C)'] = temperatura
        ys_pred = rf_reg.predict(X_reg_in)[0]
        
        # Mostrar resultados
        st.divider()
        res1, res2 = st.columns(2)
        res1.metric("Estructura Cristalina", fase_pred)
        res2.metric("Límite Elástico Estimado (YS)", f"{ys_pred:.1f} MPa")
        
        # Gráfica de probabilidades por fase
        st.subheader("Probabilidad por fase cristalina")
        df_probs = pd.DataFrame({'Fase': clases, 'Probabilidad (%)': probs * 100}).set_index('Fase')
        st.bar_chart(df_probs)
        
        # Desglose de composición normalizada
        with st.expander("Ver fracciones atómicas calculadas"):
            df_comp = pd.DataFrame({
                'Elemento': list(comp.keys()),
                'Fracción Atómica (%)': [round(v * 100, 2) for v in comp.values()]
            })
            st.dataframe(df_comp, hide_index=True, use_container_width=True)