# MadriDeep AI: Tasación Inteligente del Mercado Inmobiliario en Madrid 🏠🤖

[](https://www.google.com/search?q=https://www.python.org/)
[](https://www.google.com/search?q=https://streamlit.io/)
[](https://www.google.com/search?q=)
[](https://www.google.com/search?q=)

**MadriDeep AI** es una solución *end-to-end* diseñada para aportar transparencia al mercado inmobiliario de la Comunidad de Madrid. Mediante el uso de Inteligencia Artificial Generativa y modelos predictivos avanzados, permitimos a particulares e inversores tasar viviendas y recibir asesoría experta basada en datos reales.

-----

## 🌟 Funcionalidades Principales

  * **📊 Tasador Inteligente:** Copia y pega el texto de cualquier anuncio inmobiliario. Nuestra IA extrae automáticamente las características (metros, habitaciones, distrito, etc.) y predice el "precio justo" de mercado.
  * **💬 Asesor Conversacional:** Un chatbot especializado en el mercado madrileño impulsado por Llama 3.3, capaz de recomendar zonas de inversión y responder dudas con memoria de conversación.
  * **🛡️ Guardrails de Seguridad:** El sistema está blindado para responder exclusivamente sobre el sector inmobiliario de Madrid, garantizando un uso profesional de los recursos.

-----

## 🛠️ Stack Tecnológico

  * **Lenguaje:** Python 3.11
  * **Análisis de Datos:** Pandas, NumPy, Matplotlib, Seaborn.
  * **Modelado:** Scikit-Learn, XGBoost, TensorFlow, Keras.
  * **IA Generativa:** Llama 3.3-70b (vía Groq API).
  * **Interfaz:** Streamlit.
  * **Infraestructura:** Gestión de entornos con `uv`, versionado de modelos con `Git LFS`.

-----

## 🧠 El Desafío Técnico: ML Clásico vs. Deep Learning

Uno de los puntos clave del proyecto fue la comparativa entre algoritmos de **Machine Learning** (XGBoost/Random Forest) y una arquitectura de **Deep Learning** propia.

### La arquitectura de Redes Neuronales

Diseñé una red densa multicapa (128-64-32 neuronas) en formato `.keras`. Durante el entrenamiento, nos enfrentamos a la **"Explosión de Gradientes"** debido a la transformación inversa exponencial de los precios.
**Solución implementada:**

1.  **Batch Normalization:** Para estabilizar las activaciones internas.
2.  **Huber Loss:** Como función de pérdida robusta frente a *outliers*.
3.  **Clipping Matemático:** Para acotar lógicamente las predicciones en la fase de inferencia.

### Decisión de Ingeniería

Aunque logramos estabilizar la Red Neuronal, demostramos empíricamente que **XGBoost** ofrecía una mayor robustez y un R² superior (\~90%) para este volumen de datos tabulares.

> **Nota de Pragmátismo:** Dado el compromiso con la estabilidad en producción y los plazos de entrega, priorizamos el despliegue de los modelos de ML clásicos como motor definitivo, manteniendo el Deep Learning como un *baseline* avanzado de investigación. **Entregar un producto funcional y escalable (MVP) fue nuestra prioridad.**

-----

## 📁 Estructura del Repositorio

```text
├── 01_data_source      # Dataset original (Kaggle/Portales)
├── 02_datasets         # Datos limpios y preparados para entrenamiento
├── 04_dl_notebooks     # Investigación y entrenamiento de Redes Neuronales
├── 05_ml_notebooks     # Modelado con XGBoost y Random Forest
├── ai_assistant        # Código fuente de la aplicación Streamlit (app.py)
├── Documentacion       # Memoria técnica y presentaciones
└── .streamlit          # Configuración de secretos (API Keys)
```

-----

## ⚙️ Instalación y Uso Local

Este proyecto utiliza **`uv`** para una gestión de dependencias ultra-rápida.

1.  **Clonar el repositorio:**

    ```bash
    git clone https://github.com/Eljhansito/MadriDeep-AI-Project.git
    cd MadriDeep-AI
    ```

2.  **Instalar dependencias:**

    ```bash
    uv sync
    uv pip install -e .
    ```

3.  **Configurar API Key:** Crea un archivo `.streamlit/secrets.toml`:

    ```toml
    GROQ_API_KEY = "tu_api_key_aqui"
    ```

4.  **Ejecutar la App:**

    ```bash
    streamlit run ai_assistant/app.py
    ```

-----

## 🔮 Próximos Pasos (Roadmap)

  * **Real-time Scraping:** Integración de un pipeline de ingesta diaria para superar el uso de datasets estáticos.
  * **Computer Vision:** Modelo para evaluar el estado de conservación de la vivienda a través de las fotos del anuncio.
  * **Módulo Financiero:** Calculadora de hipotecas y ROI para inversores integrada en el chat.

-----

## 👥 Equipo

  * **Jhan Franco Schotborgh** - *Deep Learning & Arquitectura de Integración*
  * **Alvar Garcia** - *Machine Learning & Modelado*
  * **Alfredo Naranjo** - *Data Engineering & EDA*
  * **Rocío Sánchez** - *Frontend & Streamlit*

-----

## 🤝 Contribuciones

Las sugerencias y mejoras son bienvenidas! Siéntete libre de clonar el repositorio, abrir un issue o enviar un pull request para seguir mejorando MadriDeep AI.

-----

## 🚀 Demo en Vivo

Puedes probar la aplicación funcionando en tiempo real aquí:
👉 **[Demo](https://proyectofinal-py3fygv6s4oynpsdkkoqcv.streamlit.app/)**

> ⚠️ **Aviso sobre la disponibilidad:** Esta demo pública utiliza una API Key provisional habilitada exclusivamente para la defensa académica del proyecto, por lo que estará **activa por tiempo limitado**. 
> 
> Si llegas aquí y la demo ya no está disponible, o si deseas usar la herramienta sin restricciones, ¡te animamos a clonar este repositorio y ejecutarlo en local! Solo necesitas crear tu propia clave gratuita en Groq siguiendo las instrucciones de instalación.
