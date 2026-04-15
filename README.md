# MadriDeep AI
## Tasación Inteligente del Mercado Inmobiliario
### Bootcamp KeepCoding
### Big Data 16

**Autores**:
- Alfredo Naranjo Serres
- Alvar Garcia Moral
- Jhan Franco Schotborgh Piersanti
- María del Rocío Sánchez Quintana

### Descripción general del contenido de los directorios:
- El directorio **01_data_source** contiene el archivo original usado en el análisis.
- El directorio **02_datasets** contiene los archivos usados para separación, limpieza, completado y preparación para modelos de ML y DL.
- Los directorios **04_dl_notebooks** y **05_ml_notebooks** contienen la información de los notebooks que generan los respectivos modelos predictivos y el registro en MLFlow.
- El directorio **ai_assistant** tiene script usado para ejecutar la aplicación en streamlit.
- El directorio **Documentacion** contiene la memoria del proyecto y otros relacionados a su presentación.

### Consideraciones:
- El proyecto se ha probado en python 3.11.
- Se ha usado el gestor de proyectos **uv**. Para instalar las dependencias basta con instalar uv y ejecutar el comando **uv sync** en el terminal con el entorno virtual activado. (Se debe haber ejecutado también el comando uv pip install -e .).
- Para ejecutar en local se debe crear un directorio **.streamlit** y dentro del mismo debe crearse un archivo **secrets.toml** que contenga la GROQ API KEY creada en el website de Groq.
- Para ejecución en local el comando desde el directorio raiz es **streamlit run ai_assistant/app.py**.

- Se debe activar Git Large File para hacer push de archivos grandes. (Este fue el caso de los modelos de ML generados). [Git LFS](https://git-lfs.com/)


Para [Demo](https://proyectofinal-py3fygv6s4oynpsdkkoqcv.streamlit.app/)