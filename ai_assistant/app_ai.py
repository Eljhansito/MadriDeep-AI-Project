import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from groq import Groq
import json 
import mlflow.pyfunc

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(page_title="MadriDeep AI (MLflow)", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    .big-title { font-size:40px !important; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 10px; }
    .sub-title { font-size:20px !important; color: #6B7280; text-align: center; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE RECURSOS (MLflow & Preprocessor)
# ==========================================
@st.cache_resource
def cargar_recursos():
    try:
        # Localización de archivos relativa a este script
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Carga del preprocesador desde carpeta tools
        ruta_prep = os.path.join(base_path, '../tools/preprocessor.joblib')
        prep = joblib.load(ruta_prep)
        
        # Rutas a artefactos de MLflow 
        ruta_buy = os.path.join(base_path, "../mlruns/708739769633530405/0a9b06e3c3a946db984c51a5b4869808/artifacts/modelo_ml_campeon")
        ruta_rent = os.path.join(base_path, "../mlruns/708739769633530405/0d37db62e4354a7d8f00a6ea5d506d8a/artifacts/champion_ml_model")

        # Carga universal de modelos
        mod_buy = mlflow.pyfunc.load_model(ruta_buy)
        mod_rent = mlflow.pyfunc.load_model(ruta_rent)
        
        return prep, mod_buy, mod_rent, True
    except Exception as e:
        st.error(f"Error en carga de recursos: {e}")
        return None, None, None, False

preprocessor, m_buy, m_rent, modelos_listos = cargar_recursos()

# Cliente Groq
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Error: GROQ_API_KEY no configurada en Secrets")

# ==========================================
# 3. LÓGICA DE ASISTENTE (Chat)
# ==========================================
def hablar_con_ia(mensaje_usuario, contexto):
    try:
        instrucciones = (
            f"Eres 'MadriDeep', experto inmobiliario en Madrid. Contexto: {contexto}. "
            "Solo respondes sobre el mercado de Madrid. Otros temas quedan fuera de tu alcance."
        )
        mensajes_ia = [{"role": "system", "content": instrucciones}]
        if "messages" in st.session_state:
            for m in st.session_state.messages:
                mensajes_ia.append({"role": m["role"], "content": m["content"]})
        mensajes_ia.append({"role": "user", "content": mensaje_usuario})
        
        completion = client.chat.completions.create(
            messages=mensajes_ia,
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error en consulta: {str(e)}"

# ==========================================
# 4. EXTRACCIÓN Y PREDICCIÓN (Pipeline)
# ==========================================
def extraer_y_predecir(texto_anuncio):
    prompt_sistema = """
    Analiza el texto y determina si es un anuncio inmobiliario.
    Devuelve un JSON con: 'es_anuncio_inmobiliario' (bool).
    Si es true, incluye: sq_mt_built, n_rooms, n_bathrooms, floor, house_type_id, Distrito, neighborhood_id, subtitle, energy_certificate, is_exterior, is_renewal_needed, is_floor_under, has_lift, has_parking, is_new_development.
    """
    try:
        respuesta = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt_sistema}, {"role": "user", "content": texto_anuncio}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        diccionario_ia = json.loads(respuesta.choices[0].message.content)

        if not diccionario_ia.get("es_anuncio_inmobiliario", True):
            return "NO_ES_ANUNCIO", None, None

        # Valores por defecto para el preprocesador
        DEFAULTS = {'sq_mt_built': 80.0, 'n_rooms': 3.0, 'n_bathrooms': 1.0, 'floor': 1.0, 'is_exterior': False, 'is_renewal_needed': False, 'is_floor_under': False, 'has_lift': False, 'has_parking': False, 'is_new_development': False, 'house_type_id': 'Pisos', 'energy_certificate': 'other', 'subtitle': 'Centro', 'Distrito': 'Centro', 'neighborhood_id': 'Centro'}
        
        diccionario_seguro = {col: diccionario_ia.get(col, DEFAULTS.get(col)) for col in DEFAULTS.keys()}
        
        # Transformación y Predicción
        df = pd.DataFrame([diccionario_seguro])
        datos_procesados = preprocessor.transform(df)
        
        # Modelo Compra (Log-transform)
        pred_compra = m_buy.predict(datos_procesados)
        precio_compra = np.exp(pred_compra[0]) 
        
        # Modelo Alquiler
        precio_alquiler = m_rent.predict(datos_procesados)[0]
        
        return diccionario_seguro, precio_compra, precio_alquiler
    except Exception as e:
        st.error(f"Error en pipeline: {e}") 
        return None, None, None

# ==========================================
# 5. INTERFAZ DE USUARIO (Streamlit)
# ==========================================
with st.sidebar:
    st.title("🏢 MadriDeep Pro")
    if modelos_listos:
        st.success("✅ MLflow: Modelos Cargados")
    else:
        st.error("❌ MLflow: Error de conexión")

st.markdown('<p class="big-title">MadriDeep AI</p>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["💬 Chat Asesor", "📊 Tasador Automático"])

with tab1:
    st.markdown('<p class="sub-title">Consultoría inmobiliaria especializada en Madrid</p>', unsafe_allow_html=True)
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    if prompt := st.chat_input("¿Cuál es la tendencia en Chamberí?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            r = hablar_con_ia(prompt, "General")
            st.markdown(r)
            st.session_state.messages.append({"role": "assistant", "content": r})

with tab2:
    st.markdown('<p class="sub-title">Predicción de precios mediante modelos de Deep Learning</p>', unsafe_allow_html=True)
    texto = st.text_area("Inserte el contenido del anuncio:", height=150)
    
    if st.button("🚀 Ejecutar Tasación", type="primary"):
        with st.spinner("Procesando..."):
            datos, p_compra, p_alquiler = extraer_y_predecir(texto)
            if datos == "NO_ES_ANUNCIO":
                st.warning("El texto no ha sido identificado como un anuncio inmobiliario.")
            elif datos:
                c1, c2 = st.columns(2)
                c1.metric("Venta Estimada", f"{p_compra:,.0f} €")
                c2.metric("Alquiler Estimado", f"{p_alquiler:,.0f} €/mes")
                st.json(datos)