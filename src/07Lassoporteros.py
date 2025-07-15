import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

def analizar_porteros_lasso(archivo_path="DB_separadas/07_db_porteros.csv", porcentaje_mantener=40):
    """
    Analiza la importancia de variables usando LASSO para predecir el valor de mercado de porteros,
    identifica el bottom 60% de atributos menos importantes y los elimina del archivo original
    """
    
    print("🥅 ANÁLISIS LASSO PARA PORTEROS")
    print(f"📊 Manteniendo el {porcentaje_mantener}% de features más relevantes")
    print(f"🗑️ Eliminando el {100-porcentaje_mantener}% de features menos importantes")
    print("="*60)
    
    # Variables predictoras que SÍ participan en LASSO
    variables_lasso = [
        'Pie bueno', 'understat_matches', 'understat_minutes', 'overallrating', 
        'potential', 'crossing', 'finishing', 'headingaccuracy', 'shortpassing',
        'volleys', 'dribbling', 'curve', 'fk_accuracy', 'longpassing', 
        'ballcontrol', 'acceleration', 'sprintspeed', 'agility', 'reactions', 
        'balance', 'shotpower', 'jumping', 'stamina', 'strength', 'longshots',
        'aggression', 'interceptions', 'positioning', 'vision', 'penalties', 
        'composure', 'defensiveawareness', 'standingtackle', 'slidingtackle', 
        'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes'
    ]
    
    # Variable objetivo
    variable_objetivo = 'Valor de mercado actual (numérico)'
    
    try:
        # 📂 CARGAR DATOS
        print(f"📂 Cargando datos desde: {archivo_path}")
        df = pd.read_csv(archivo_path)
        print(f"   Dimensiones originales: {df.shape}")
        
        # 🔍 ANÁLISIS DE VARIABLES DISPONIBLES
        print(f"\n🔍 ANÁLISIS DE VARIABLES EN EL DATASET:")
        todas_las_columnas = list(df.columns)
        
        # Variables que SÍ participan en LASSO (disponibles)
        variables_lasso_disponibles = [var for var in variables_lasso if var in df.columns]
        variables_lasso_faltantes = [var for var in variables_lasso if var not in df.columns]
        
        # Variables que NO participan en LASSO (se mantienen automáticamente)
        variables_no_lasso = [col for col in todas_las_columnas 
                             if col not in variables_lasso and col != variable_objetivo]
        
        print(f"\n📊 RESUMEN DE VARIABLES:")
        print(f"   Total de columnas en dataset: {len(todas_las_columnas)}")
        print(f"   Variables para análisis LASSO: {len(variables_lasso_disponibles)}")
        print(f"   Variables NO analizadas por LASSO (se mantienen): {len(variables_no_lasso)}")
        print(f"   Variable objetivo: 1")
        
        print(f"\n✅ VARIABLES ANALIZADAS POR LASSO ({len(variables_lasso_disponibles)}):")
        for i, var in enumerate(variables_lasso_disponibles, 1):
            print(f"   {i:2d}. {var}")
        
        print(f"\n🔒 VARIABLES NO ANALIZADAS (SE MANTIENEN AUTOMÁTICAMENTE):")
        if len(variables_no_lasso) > 0:
            for i in range(0, len(variables_no_lasso), 5):
                grupo = variables_no_lasso[i:i+5]
                print(f"   {', '.join(grupo)}")
        else:
            print("   (Ninguna)")
        
        # 🔍 VERIFICAR DATOS
        if variable_objetivo not in df.columns:
            raise ValueError(f"La variable objetivo '{variable_objetivo}' no se encuentra en el dataset")
        
        print(f"\n📊 INFORMACIÓN DEL DATASET:")
        print(f"   Total de porteros: {len(df)}")
        print(f"   Valor de mercado - Min: ${df[variable_objetivo].min():,.0f}")
        print(f"   Valor de mercado - Max: ${df[variable_objetivo].max():,.0f}")
        print(f"   Valor de mercado - Promedio: ${df[variable_objetivo].mean():,.0f}")
        
        # 🧹 PREPARAR DATOS PARA LASSO
        print(f"\n🧹 PREPARANDO DATOS PARA LASSO:")
        
        # Seleccionar solo variables para LASSO
        X = df[variables_lasso_disponibles].copy()
        y = df[variable_objetivo].copy()
        
        # Eliminar filas con valores faltantes en la variable objetivo
        mask_y_valido = ~y.isna()
        X = X[mask_y_valido]
        y = y[mask_y_valido]
        
        print(f"   Después de eliminar valores faltantes en objetivo: {len(X)} muestras")
        
        if len(X) == 0:
            raise ValueError("No hay datos válidos después de la limpieza")
        
        # Manejar valores faltantes en las variables predictoras
        print(f"   Valores faltantes por variable:")
        missing_info = X.isnull().sum()
        missing_info = missing_info[missing_info > 0]
        if len(missing_info) > 0:
            for var, count in missing_info.items():
                print(f"     {var}: {count} ({count/len(X)*100:.1f}%)")
            X = X.fillna(X.median())
        else:
            print(f"     No hay valores faltantes")
        
        # Verificar y manejar valores infinitos
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        print(f"   Dataset final para LASSO: {len(X)} porteros, {len(variables_lasso_disponibles)} variables")
        
        if len(X) < 10:
            raise ValueError("Muy pocas muestras para realizar el análisis")
        
        # 🎯 EJECUTAR ANÁLISIS LASSO
        print(f"\n🎯 EJECUTANDO ANÁLISIS LASSO:")
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Datos de entrenamiento: {len(X_train)} muestras")
        print(f"   Datos de prueba: {len(X_test)} muestras")
        
        # Estandarizar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Configurar LASSO con validación cruzada
        alphas = np.logspace(-6, 2, 100)
        lasso_cv = LassoCV(
            alphas=alphas, 
            cv=5,
            random_state=42, 
            max_iter=5000,
            n_jobs=-1
        )
        
        # Entrenar modelo
        print(f"   Entrenando modelo LASSO con validación cruzada...")
        lasso_cv.fit(X_train_scaled, y_train)
        
        # Obtener resultados
        coeficientes = lasso_cv.coef_
        
        # Crear DataFrame con resultados
        resultados_lasso = pd.DataFrame({
            'Variable': variables_lasso_disponibles,
            'Coeficiente': coeficientes,
            'Importancia_Abs': np.abs(coeficientes)
        })
        
        # Ordenar por importancia absoluta
        resultados_lasso = resultados_lasso.sort_values('Importancia_Abs', ascending=False)
        
        # 🔍 IDENTIFICAR TOP Y BOTTOM FEATURES
        print(f"\n🔍 IDENTIFICANDO FEATURES A MANTENER Y ELIMINAR:")
        
        # Calcular número de features a mantener (top %)
        num_features_total = len(variables_lasso_disponibles)
        num_features_mantener = max(1, int(num_features_total * porcentaje_mantener / 100))
        num_features_eliminar = num_features_total - num_features_mantener
        
        print(f"   Features analizadas por LASSO: {num_features_total}")
        print(f"   Features a MANTENER (top {porcentaje_mantener}%): {num_features_mantener}")
        print(f"   Features a ELIMINAR (bottom {100-porcentaje_mantener}%): {num_features_eliminar}")
        
        # Seleccionar features
        top_features = resultados_lasso.head(num_features_mantener)['Variable'].tolist()
        # Para bottom features: saltar las top y tomar el resto
        bottom_features = resultados_lasso.iloc[num_features_mantener:]['Variable'].tolist()
        
        print(f"\n🏆 TOP {porcentaje_mantener}% FEATURES (SE MANTIENEN):")
        for i, feature in enumerate(top_features, 1):
            importancia = resultados_lasso[resultados_lasso['Variable'] == feature]['Importancia_Abs'].iloc[0]
            print(f"   {i:2d}. {feature} (importancia: {importancia:.4f})")
        
        print(f"\n🗑️ BOTTOM {100-porcentaje_mantener}% FEATURES (SE ELIMINAN):")
        if len(bottom_features) > 0:
            for i, feature in enumerate(bottom_features, 1):
                importancia = resultados_lasso[resultados_lasso['Variable'] == feature]['Importancia_Abs'].iloc[0]
                print(f"   {i:2d}. {feature} (importancia: {importancia:.4f})")
        else:
            print("   (Ninguna)")
        
        # 🗑️ CREAR ARCHIVO FILTRADO
        print(f"\n🗑️ CREANDO ARCHIVO FILTRADO (ELIMINANDO BOTTOM {100-porcentaje_mantener}%):")
        
        # Columnas finales: variables no analizadas + top features + variable objetivo
        columnas_mantener = variables_no_lasso + top_features + [variable_objetivo]
        
        # Crear DataFrame filtrado
        df_filtrado = df[columnas_mantener].copy()
        
        # Guardar archivo filtrado
        nombre_archivo_original = os.path.basename(archivo_path)
        nombre_sin_extension = nombre_archivo_original.replace('.csv', '')
        output_path = f"DB_separadas/08_db_porteros_filtered_top{porcentaje_mantener}pct.csv"
        
        df_filtrado.to_csv(output_path, index=False)
        
        print(f"   ✅ Archivo filtrado creado: {output_path}")
        print(f"   📊 Dimensiones originales: {df.shape}")
        print(f"   📊 Dimensiones filtradas: {df_filtrado.shape}")
        print(f"   🔢 Columnas eliminadas: {df.shape[1] - df_filtrado.shape[1]}")
        print(f"   🔢 Columnas mantenidas: {df_filtrado.shape[1]}")
        
        # Desglose detallado
        print(f"\n📋 DESGLOSE DE COLUMNAS:")
        print(f"   Variables no analizadas (mantenidas): {len(variables_no_lasso)}")
        print(f"   Top {porcentaje_mantener}% features (mantenidas): {len(top_features)}")
        print(f"   Bottom {100-porcentaje_mantener}% features (eliminadas): {len(bottom_features)}")
        print(f"   Variable objetivo (mantenida): 1")
        print(f"   Total mantenidas: {len(columnas_mantener)}")
        
        # 💾 GUARDAR ANÁLISIS
        print(f"\n💾 GUARDANDO ANÁLISIS:")
        
        # Crear carpeta lasso si no existe
        os.makedirs('lasso', exist_ok=True)
        
        # Guardar ranking de importancia
        output_ranking = f"lasso/08_ranking_importancia_porteros_top{porcentaje_mantener}pct.csv"
        resultados_lasso.to_csv(output_ranking, index=False)
        print(f"   Ranking de importancia: {output_ranking}")
        
        # Guardar lista de features eliminadas
        if len(bottom_features) > 0:
            df_eliminadas = pd.DataFrame({
                'Feature_Eliminada': bottom_features,
                'Importancia': [resultados_lasso[resultados_lasso['Variable'] == f]['Importancia_Abs'].iloc[0] 
                               for f in bottom_features]
            })
            output_eliminadas = f"lasso/08_features_eliminadas_porteros_bottom{100-porcentaje_mantener}pct.csv"
            df_eliminadas.to_csv(output_eliminadas, index=False)
            print(f"   Features eliminadas: {output_eliminadas}")
        
        # Métricas del modelo
        y_pred = lasso_cv.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n📈 MÉTRICAS DEL MODELO LASSO:")
        print(f"   Alpha óptimo: {lasso_cv.alpha_:.6f}")
        print(f"   R² en prueba: {r2:.4f}")
        print(f"   RMSE en prueba: ${rmse:,.0f}")
        
        # 🎯 RESUMEN FINAL
        print(f"\n🎯 RESUMEN FINAL:")
        print(f"   ✅ Archivo original: {archivo_path}")
        print(f"   ✅ Archivo filtrado: {output_path}")
        print(f"   📉 Reducción de columnas: {df.shape[1]} → {df_filtrado.shape[1]} ({df.shape[1] - df_filtrado.shape[1]} eliminadas)")
        print(f"   🎯 Criterio: Eliminar bottom {100-porcentaje_mantener}% de features según importancia LASSO")
        
        return df_filtrado, resultados_lasso, top_features, bottom_features
        
    except FileNotFoundError:
        print(f"❌ ERROR: No se pudo encontrar el archivo {archivo_path}")
        return None, None, None, None
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None, None, None, None

# Función adicional para visualización
def crear_graficos_importancia(resultados_lasso, save_plots=True, porcentaje_mantener=40):
    """
    Crea gráficos de la importancia de variables mostrando solo las features seleccionadas
    """
    if resultados_lasso is None:
        print("No hay resultados para graficar")
        return
    
    # Filtrar solo las variables con importancia > 0 para el gráfico
    vars_importantes = resultados_lasso[resultados_lasso['Importancia_Abs'] > 0].copy()
    
    if len(vars_importantes) == 0:
        print("No hay variables con importancia > 0 para graficar")
        return
    
    # Tomar solo las top features para el gráfico (máximo 20 para legibilidad)
    max_vars_grafico = min(20, len(vars_importantes))
    vars_importantes = vars_importantes.head(max_vars_grafico)
    
    # Calcular altura necesaria basada en el número de variables
    altura_por_variable = 0.5  # altura en pulgadas por variable
    altura_total = max(8, len(vars_importantes) * altura_por_variable)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, altura_total))
    
    # Gráfico 1: Importancia absoluta
    ax1.barh(range(len(vars_importantes)), vars_importantes['Importancia_Abs'])
    ax1.set_yticks(range(len(vars_importantes)))
    ax1.set_yticklabels(vars_importantes['Variable'])
    ax1.set_xlabel('Importancia Absoluta')
    ax1.set_title(f'Top {max_vars_grafico} Variables - Importancia Absoluta\n(Análisis LASSO para Porteros - Manteniendo {porcentaje_mantener}%)')
    ax1.grid(axis='x', alpha=0.3)
    
    # Gráfico 2: Coeficientes (con signo)
    colors = ['red' if x < 0 else 'blue' for x in vars_importantes['Coeficiente']]
    ax2.barh(range(len(vars_importantes)), vars_importantes['Coeficiente'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(vars_importantes)))
    ax2.set_yticklabels(vars_importantes['Variable'])
    ax2.set_xlabel('Coeficiente LASSO')
    ax2.set_title(f'Top {max_vars_grafico} Variables - Coeficientes LASSO\n(Rojo: Negativo, Azul: Positivo)')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Ajustar espaciado
    plt.tight_layout()
    
    if save_plots:
        # Crear carpeta lasso si no existe
        os.makedirs('lasso', exist_ok=True)
        plt.savefig(f'lasso/08_importancia_variables_porteros_top{porcentaje_mantener}pct.png', dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado: lasso/08_importancia_variables_porteros_top{porcentaje_mantener}pct.png")
    
    plt.show()

# Ejecutar análisis principal
if __name__ == "__main__":
    print("🚀 INICIANDO ANÁLISIS LASSO PARA PORTEROS - ELIMINANDO BOTTOM 60% DE FEATURES")
    print("="*60)
    
    # Ejecutar análisis con 40% de features más relevantes (eliminar bottom 60%)
    df_filtrado, resultados_lasso, top_features, bottom_features = analizar_porteros_lasso(porcentaje_mantener=40)
    
    # Crear gráficos si hay resultados
    if resultados_lasso is not None:
        print(f"\n📊 Creando gráficos de importancia...")
        crear_graficos_importancia(resultados_lasso, porcentaje_mantener=40)
    
    print(f"\n✅ ANÁLISIS COMPLETADO")
    print("="*60)