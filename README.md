Para visualizar el dashboard: 
- streamlit run main.py


# Wine Dataset Visualization
Este proyecto es una aplicación interactiva desarrollada con Streamlit que permite visualizar y analizar un conjunto de datos de vinos (winemag-data-130k-v2.csv). La aplicación ofrece información geográfica, detalles sobre bodegas, análisis de precios y distribuciones, además de permitir filtrar los datos por país y bodega.

# Objetivo
Esta proyecto fue desarrollado como parte de un trabajo universitario, y su objetivo principal es proporcionar una herramienta visual e interactiva para explorar un conjunto de datos de vinos, analizando aspectos como:
- Producción por continente, país y provincia.
- Comparativas entre países (puntuación promedio, precio promedio y relación precio/puntuación).
- Información sobre bodegas (puntuación, producción y análisis de sentimientos).
- Distribución de precios y su relación con otras variables como la edad y los puntos.

# Estructura del Proyecto
El proyecto está dividido en dos archivos principales:

1. main.py: Contiene la lógica principal de la aplicación Streamlit, incluyendo la carga de datos y la estructura de la interfaz de usuario.
2. functions.py: Incluye todas las funciones auxiliares para el procesamiento de datos, imputación, análisis de sentimientos y visualizaciones.

# Uso
Una vez que la aplicación está en ejecución, se abrirá en tu navegador predeterminado. La interfaz está dividida en las siguientes secciones:

1. Vista Previa del Conjunto de Datos: Muestra el DataFrame completo procesado.
2. Información Geográfica:
- Producción por continente, países y provincias.
- Países con precios promedio más altos.
- Países con la mejor relación precio/puntuación.
- Comparación entre países (puntuación, precio y relación).
- Filtro interactivo por país.
3. Sobre las Bodegas:
- Comparación de bodegas por puntuación promedio, relación precio/puntuación y cantidad de producción.
- Bodegas con más reseñas positivas (basado en análisis de sentimientos).
- Filtro interactivo por bodega.
4. Sobre el Precio:
- Distribución de precios (filtrada hasta el decil 0.9).
- Relación entre precio, edad y puntos (análisis de correlación y gráfico de dispersión).

# Procesamiento de Datos
El conjunto de datos se carga desde una URL pública y se procesa con las siguientes transformaciones:
- Eliminación de duplicados y valores nulos en columnas críticas (country, description).
- Asignación de continentes basada en el país.
- Imputación de valores faltantes en price y year usando métodos personalizados.
- Cálculo de la edad del vino (2024 - year).
- Análisis de sentimientos en las descripciones.
- Creación de una columna points vs price para evaluar la relación precio/puntuación.



