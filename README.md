# 🏆 Predictor de Valor de Mercado de Jugadores de Fútbol

Este proyecto implementa un sistema completo de análisis y predicción del valor de mercado de jugadores de fútbol utilizando técnicas de Machine Learning, incluyendo clustering K-means, reducción de dimensionalidad con LASSO, modelos predictivos avanzados como XGBoost y SVR, y **redes neuronales** para predicción especializada por posición.

## 📁 Estructura del Proyecto

### 📊 Datos/
Contiene las bases de datos procesadas en diferentes etapas:

- **DB_unidas/**: Bases de datos procesadas secuencialmente por los scripts numerados
  - `01_df_filtrado_final.csv` - Dataset inicial filtrado
  - `02_df_columnas_eliminadas.csv` - Tras eliminación de columnas irrelevantes  
  - `03_db_columnas_ordenadas.csv` - Con columnas reorganizadas
  - `04_db_codificado.csv` - Variables categóricas codificadas
  - `05_db_normalizado.csv` - Datos normalizados
  - `06_db_completo.csv` - Dataset completo procesado
  - `06.5_db_portero.csv` - Dataset específico para porteros

- **DB_separadas/**: Bases de datos divididas por posición tras aplicar filtros del top 40%
  - `09_db_centrocampista_filtered_top40pct.csv`
  - `09_db_porteros_filtered_top40pct.csv` 
  - `09_db_defensa_filtered_top40pct.csv`
  - `09_db_delantero_filtered_top40pct.csv`

### 💻 src/
Códigos de procesamiento de datos y modelado que deben ejecutarse **en orden numérico**:

#### 🔧 Procesamiento de Datos (01-08):
1. `01eliminadordecolumnas.py` - Elimina columnas irrelevantes
2. `02ordenadordecolumnas.py` - Reorganiza las columnas del dataset
3. `03onehotencoder.py` - Codifica variables categóricas
4. `04normalizador.py` - Normaliza los datos numéricos
5. `05completador.py` - Completa valores faltantes
6. `06separador.py` - Separa datos por posiciones
7. `06z5limpiezaporteros.py` - Procesamiento específico para porteros
8. `07Lasso.py` - Reduce dimensionalidad con LASSO para jugadores de campo
9. `07Lassoporteros.py` - Reduce dimensionalidad con LASSO para porteros
10. `08kmeans.py` - Aplica clustering K-means para clasificar jugadores

#### 🤖 Modelos Predictivos Clásicos:
- `XGBoost.py` - Modelo XGBoost con clusters
- `XGBoost_no_clusters.py` - Modelo XGBoost sin clusters
- `SVR.py` - Support Vector Regression con clusters
- `SVR_no_clusters.py` - Support Vector Regression sin clusters

#### 🧠 Modelos de Redes Neuronales (Jupyter Notebooks):
- `porterosRN.ipynb` - Redes neuronales especializadas para porteros
- `defensaRN.ipynb` - Redes neuronales especializadas para defensas
- `centrocampistaRN.ipynb` - Redes neuronales especializadas para centrocampistas
- `delanterosRN.ipynb` - Redes neuronales especializadas para delanteros
- `generalRN.ipynb` - Redes neuronales generales (todos los jugadores)

#### 📊 Análisis Final:
- `analisis_final.py` - Análisis exploratorio completo y generación de reportes finales

### 📈 Resultados/
Gráficas y análisis de resultados organizados por técnica:
- **Lasso/** - Resultados de reducción de dimensionalidad
- **XGBoost/** - Métricas y visualizaciones del modelo XGBoost
- **SVR/** - Métricas y visualizaciones del modelo SVR
- **k-means/** - Análisis de clustering
- **Redes_Neuronales/** - Resultados de los modelos de deep learning

### 🌐 web/
Aplicación web interactiva para visualizar y utilizar los modelos:
- `pagina.html` - Interfaz principal de la aplicación
- `pagina.css` - Estilos de la aplicación
- `pagina.js` - Lógica de interacción y visualizaciones
- `data.csv` - Dataset con las mejores predicciones para la web
- **data_preparation/** - Scripts de preparación de datos para la web

### 📋 Otros Archivos
- `requirements.txt` - Dependencias del proyecto
- `Docs/` - Documentación adicional
- `.venv/` - Entorno virtual de Python

## 🚀 Instalación y Configuración

### 1. Configurar el Entorno
```bash
# Clonar o descargar el proyecto
cd Codigo-TFM-K-means

# Crear entorno virtual (si no existe)
python -m venv .venv

# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Dependencias Principales
- pandas>=1.3.0
- scikit-learn==1.3.2
- numpy>=1.21.0
- matplotlib==3.8.0
- seaborn==0.13.0
- scikit-optimize
- scipy
- xgboost
- **tensorflow>=2.0** (para redes neuronales)
- **keras** (para redes neuronales)

## 📝 Ejecución de los Códigos

### Procesamiento de Datos
Los scripts en `src/` deben ejecutarse **en orden numérico** para procesar correctamente los datos:

```bash
cd src/

# Procesamiento secuencial (OBLIGATORIO en este orden)
python 01eliminadordecolumnas.py
python 02ordenadordecolumnas.py  
python 03onehotencoder.py
python 04normalizador.py
python 05completador.py
python 06separador.py
python 06z5limpiezaporteros.py
python 07Lasso.py
python 07Lassoporteros.py
python 08kmeans.py
```

### Entrenamiento de Modelos

#### Modelos Clásicos
Una vez procesados los datos, ejecutar los modelos predictivos:

```bash
# Modelos con clustering
python XGBoost.py
python SVR.py

# Modelos sin clustering (para comparación)
python XGBoost_no_clusters.py
python SVR_no_clusters.py
```

#### Modelos de Redes Neuronales
Los notebooks de redes neuronales deben ejecutarse en **Google Colab** o **Jupyter Notebook**:

```bash
# Instalar Jupyter si no está instalado
pip install jupyter

# Ejecutar Jupyter Notebook
jupyter notebook
```

**Notebooks disponibles:**
- `porterosRN.ipynb` - Modelos especializados para porteros con TensorFlow/Keras
- `defensaRN.ipynb` - Modelos especializados para defensas
- `centrocampistaRN.ipynb` - Modelos especializados para centrocampistas  
- `delanterosRN.ipynb` - Modelos especializados para delanteros
- `generalRN.ipynb` - Modelos generales para todos los jugadores

> **Nota**: Los notebooks están optimizados para Google Colab y incluyen early stopping, dropout y regularización para evitar overfitting.

### Análisis Final
```bash
# Ejecutar análisis exploratorio completo
python analisis_final.py
```

## 🌐 Aplicación Web

### Arrancar la Aplicación Web

La aplicación web es un cliente estático que puedes ejecutar así:

```bash
cd web/
python -m http.server 8000
```
Luego abre tu navegador en: `http://localhost:8000/pagina.html`

### Funcionalidades de la Web

La aplicación web te permite:

🔍 **Búsqueda Inteligente**:
- Buscar jugadores por nombre, equipo o características
- Ejemplos: "Daniel Ceballos Fernández", "Manchester City", "overallrating > 90"

📊 **Visualización Interactiva**: 
- Tabla completa de jugadores con filtros avanzados
- Filtros por valor actual y predicho
- Filtros por posición
- Ordenamiento por múltiples criterios

📈 **Análisis Detallado**:
- Modal con estadísticas detalladas de cada jugador
- Gráfico radar con atributos del jugador
- Comparación entre valor actual y predicho

### Datos de la Web

El archivo `web/data.csv` contiene:
- **Datos originales** del jugador (nombre, edad, posición, equipo, etc.)
- **Predicciones** del mejor modelo obtenido
- **Métricas** de rendimiento
- **Información de clustering** cuando aplique

## 🔬 Metodología del Proyecto

### 1. Procesamiento de Datos
- **Limpieza**: Eliminación de columnas irrelevantes y valores faltantes
- **Codificación**: Transformación de variables categóricas
- **Normalización**: Estandarización de variables numéricas
- **Segmentación**: División por posiciones de juego

### 2. Reducción de Dimensionalidad
- **LASSO Regression**: Selección automática de características más relevantes
- **Tratamiento específico**: Diferentes modelos para porteros vs jugadores de campo

### 3. Clustering
- **K-means**: Agrupación de jugadores con características similares
- **Optimización**: Determinación del número óptimo de clusters

### 4. Modelos Predictivos

#### Modelos Clásicos
- **XGBoost**: Gradient boosting para predicción robusta
- **SVR**: Support Vector Regression para relaciones no lineales
- **Comparación**: Modelos con y sin información de clustering

#### Modelos de Deep Learning
- **Redes Neuronales**: Modelos especializados por posición usando TensorFlow/Keras
- **Arquitectura**: Capas densas con dropout y regularización
- **Optimización**: Early stopping para evitar overfitting
- **Segmentación**: Modelos específicos para cada posición y cluster

### 5. Evaluación
- **Métricas**: MAE, MSE, R², RMSE
- **Validación cruzada**: Evaluación robusta del rendimiento
- **Comparación de modelos**: Análisis de rendimiento entre técnicas clásicas y deep learning
- **Visualizaciones**: Gráficas de residuos y predicciones

## 📊 Resultados

Los modelos desarrollados permiten:
- **Predicción precisa** del valor de mercado de jugadores
- **Identificación** de jugadores infravalorados/sobrevalorados  
- **Análisis** de factores que más influyen en el valor
- **Segmentación** inteligente de jugadores por características
- **Comparación** entre enfoques clásicos y deep learning

## 🎯 Uso Recomendado

1. **Para análisis completo**: Ejecuta todos los scripts en orden y revisa las gráficas en `Resultados/`
2. **Para modelos de deep learning**: Utiliza los notebooks en Google Colab para entrenar redes neuronales especializadas
3. **Para uso rápido**: Utiliza directamente la aplicación web con los datos ya procesados
4. **Para investigación**: Examina los diferentes modelos y sus métricas de rendimiento, comparando enfoques clásicos vs deep learning

## 🤖 Tecnologías Utilizadas

- **Python 3.x**: Lenguaje de programación principal
- **Pandas & NumPy**: Manipulación y análisis de datos
- **Scikit-learn**: Modelos de machine learning clásicos
- **XGBoost**: Gradient boosting avanzado
- **TensorFlow/Keras**: Deep learning y redes neuronales
- **Matplotlib & Seaborn**: Visualización de datos
- **Jupyter Notebooks**: Análisis interactivo y experimentación

## 🤝 Contribución

Este proyecto forma parte de un Trabajo de Fin de Máster sobre predicción de valor de mercado en fútbol utilizando técnicas avanzadas de Machine Learning, incluyendo tanto métodos clásicos como deep learning.

---

*Proyecto desarrollado como parte del TFM en análisis predictivo de jugadores de fútbol* ⚽ 