import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def crear_graficos_importancia(resultados, nombre_posicion, save_plots=True, porcentaje_features=50):
    """
    Crea gráficos de la importancia de variables para una posición específica,
    mostrando solo las features seleccionadas
    """
    if resultados is None:
        print("No hay resultados para graficar")
        return
    
    # Usar solo las variables seleccionadas
    vars_importantes = resultados.copy()
    
    if len(vars_importantes) == 0:
        print("No hay variables para graficar")
        return
    
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
    ax1.set_title(f'Top {porcentaje_features}% Variables - Importancia Absoluta\n(Análisis LASSO para {nombre_posicion})')
    ax1.grid(axis='x', alpha=0.3)
    
    # Gráfico 2: Coeficientes (con signo)
    colors = ['red' if x < 0 else 'blue' for x in vars_importantes['Coeficiente']]
    ax2.barh(range(len(vars_importantes)), vars_importantes['Coeficiente'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(vars_importantes)))
    ax2.set_yticklabels(vars_importantes['Variable'])
    ax2.set_xlabel('Coeficiente LASSO')
    ax2.set_title(f'Top {porcentaje_features}% Variables - Coeficientes LASSO\n(Rojo: Negativo, Azul: Positivo)')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Ajustar espaciado
    plt.tight_layout()
    
    if save_plots:
        # Crear carpeta lasso si no existe
        os.makedirs('lasso', exist_ok=True)
        plt.savefig(f'lasso/08_importancia_variables_{nombre_posicion.lower()}_top{porcentaje_features}pct.png', dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado: lasso/08_importancia_variables_{nombre_posicion.lower()}_top{porcentaje_features}pct.png")
    
    plt.show()

def analizar_importancia_lasso(carpeta_path="DB_separadas", porcentaje_mantener=40):
    """
    Analiza la importancia de variables usando LASSO para todos los CSVs en una carpeta,
    excluyendo el archivo de porteros, identifica el bottom 60% de atributos menos importantes
    y los elimina de los archivos originales
    """
    
    # Variables que NUNCA se eliminan (protegidas)
    variables_protegidas = [
        'overallrating',  # NUNCA eliminar overallrating
        'potential'       # También protegemos potential por ser crítico
    ]
    
    # Variables que SÍ participarán en el modelo LASSO (pero pueden ser eliminadas si no son importantes)
    variables_lasso = [
        'Posición específica', 'Posición secundaria', 'Pie bueno',
        'understat_matches', 'understat_minutes', 'understat_goals',
        'understat_xg', 'understat_assists', 'understat_xa',
        'understat_shots', 'understat_key_passes', 'understat_yellow_cards',
        'understat_red_cards',
        'crossing', 'finishing', 'headingaccuracy', 'shortpassing',
        'volleys', 'dribbling', 'curve', 'fk_accuracy',
        'longpassing', 'ballcontrol', 'acceleration', 'sprintspeed',
        'agility', 'reactions', 'balance', 'shotpower',
        'jumping', 'stamina', 'strength', 'longshots',
        'aggression', 'interceptions', 'positioning', 'vision',
        'penalties', 'composure', 'defensiveawareness', 'standingtackle',
        'slidingtackle'
    ]
    
    # Todas las variables que participan en análisis LASSO (protegidas + eliminables)
    todas_variables_lasso = variables_protegidas + variables_lasso
    
    # Variable objetivo
    variable_objetivo = 'Valor de mercado actual (numérico)'
    
    # Obtener lista de archivos CSV, excluyendo el de porteros
    archivos_csv = [f for f in os.listdir(carpeta_path) 
                   if f.endswith('.csv') and f != '07_db_porteros.csv']
    
    if not archivos_csv:
        print(f"No se encontraron archivos CSV válidos en la carpeta {carpeta_path}")
        return
    
    # Crear carpeta lasso si no existe
    os.makedirs('lasso', exist_ok=True)
    
    print(f"🚀 ANÁLISIS LASSO - ELIMINANDO BOTTOM {100-porcentaje_mantener}% DE FEATURES")
    print(f"📊 Manteniendo el {porcentaje_mantener}% de features más relevantes")
    print(f"🗑️ Eliminando el {100-porcentaje_mantener}% de features menos importantes")
    print(f"🔒 Variables PROTEGIDAS (nunca se eliminan): {', '.join(variables_protegidas)}")
    print(f"📁 Archivos filtrados → DB_separadas/")
    print(f"📁 Análisis → lasso/")
    print(f"❌ Excluyendo: 07_db_porteros.csv (se procesa con 07zLasso.py)")
    print(f"Encontrados {len(archivos_csv)} archivos CSV para analizar")
    print("Archivos a procesar:", archivos_csv)
    print("="*60)
    
    resultados_globales = {}
    archivos_filtrados = {}
    
    for archivo in archivos_csv:
        print(f"\n📊 ANÁLISIS DE: {archivo}")
        print("-" * 50)
        
        try:
            # Cargar datos
            ruta_completa = os.path.join(carpeta_path, archivo)
            df = pd.read_csv(ruta_completa)
            
            print(f"Dimensiones del dataset: {df.shape}")
            
            # 🔍 ANÁLISIS DE VARIABLES DISPONIBLES
            print(f"\n🔍 ANÁLISIS DE VARIABLES EN EL DATASET:")
            todas_las_columnas = list(df.columns)
            
            # Variables protegidas disponibles
            variables_protegidas_disponibles = [var for var in variables_protegidas if var in df.columns]
            
            # Variables LASSO disponibles (sin incluir las protegidas)
            variables_lasso_disponibles = [var for var in variables_lasso if var in df.columns]
            
            # Todas las variables para análisis LASSO
            todas_variables_lasso_disponibles = variables_protegidas_disponibles + variables_lasso_disponibles
            
            # Variables que NO participan en LASSO (se mantienen automáticamente)
            variables_no_lasso = [col for col in todas_las_columnas 
                                 if col not in todas_variables_lasso and col != variable_objetivo]
            
            print(f"\n📊 RESUMEN DE VARIABLES:")
            print(f"   Total de columnas en dataset: {len(todas_las_columnas)}")
            print(f"   Variables PROTEGIDAS (nunca se eliminan): {len(variables_protegidas_disponibles)}")
            print(f"   Variables para análisis LASSO (eliminables): {len(variables_lasso_disponibles)}")
            print(f"   Variables NO analizadas por LASSO (se mantienen): {len(variables_no_lasso)}")
            print(f"   Variable objetivo: 1")
            
            print(f"\n🔒 VARIABLES PROTEGIDAS (NUNCA SE ELIMINAN):")
            for i, var in enumerate(variables_protegidas_disponibles, 1):
                print(f"   {i:2d}. {var}")
            
            print(f"\n✅ VARIABLES ANALIZADAS POR LASSO (ELIMINABLES SEGÚN IMPORTANCIA):")
            for i, var in enumerate(variables_lasso_disponibles, 1):
                print(f"   {i:2d}. {var}")
            
            print(f"\n🔒 VARIABLES NO ANALIZADAS (SE MANTIENEN AUTOMÁTICAMENTE):")
            if len(variables_no_lasso) > 0:
                for i in range(0, len(variables_no_lasso), 5):
                    grupo = variables_no_lasso[i:i+5]
                    print(f"   {', '.join(grupo)}")
            else:
                print("   (Ninguna)")
            
            # Verificar que existe la variable objetivo
            if variable_objetivo not in df.columns:
                print(f"❌ Variable objetivo '{variable_objetivo}' no encontrada")
                continue
            
            print(f"\n📊 INFORMACIÓN DEL DATASET:")
            print(f"   Total de jugadores: {len(df)}")
            print(f"   Valor de mercado - Min: ${df[variable_objetivo].min():,.0f}")
            print(f"   Valor de mercado - Max: ${df[variable_objetivo].max():,.0f}")
            print(f"   Valor de mercado - Promedio: ${df[variable_objetivo].mean():,.0f}")
            
            # Preparar datos para LASSO
            X = df[todas_variables_lasso_disponibles].copy()
            y = df[variable_objetivo].copy()
            
            # Eliminar filas con valores faltantes en y
            mask_y_valido = ~y.isna()
            X = X[mask_y_valido]
            y = y[mask_y_valido]
            
            if len(X) == 0:
                print("❌ No hay datos válidos después de limpiar")
                continue
            
            # Manejar valores faltantes en X
            X_numerico = X.select_dtypes(include=[np.number])
            X_categorico = X.select_dtypes(exclude=[np.number])
            
            if not X_numerico.empty:
                X_numerico = X_numerico.fillna(X_numerico.median())
            
            if not X_categorico.empty:
                for col in X_categorico.columns:
                    X_categorico[col] = X_categorico[col].fillna(X_categorico[col].mode()[0] if not X_categorico[col].mode().empty else 0)
            
            X = pd.concat([X_numerico, X_categorico], axis=1)
            X = X.reindex(columns=todas_variables_lasso_disponibles)
            
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            print(f"\n🧹 LIMPIEZA DE DATOS:")
            print(f"   Datos después de limpieza: {len(X)} muestras")
            print(f"   Variables para LASSO: {len(todas_variables_lasso_disponibles)}")
            
            if len(X) < 10:
                print("❌ Muy pocas muestras para análisis")
                continue
            
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
            
            # 🎯 EJECUTAR ANÁLISIS LASSO
            print(f"\n🎯 EJECUTANDO ANÁLISIS LASSO:")
            alphas = np.logspace(-4, 2, 100)
            lasso_cv = LassoCV(
                alphas=alphas, 
                cv=5, 
                random_state=42, 
                max_iter=5000,
                n_jobs=-1
            )
            
            try:
                print("   Entrenando modelo LASSO con validación cruzada...")
                lasso_cv.fit(X_train_scaled, y_train)
                
                # Obtener resultados
                coeficientes = lasso_cv.coef_
                
                # Crear DataFrame con resultados
                resultados_lasso = pd.DataFrame({
                    'Variable': todas_variables_lasso_disponibles,
                    'Coeficiente': coeficientes,
                    'Importancia_Abs': np.abs(coeficientes)
                })
                
                # Ordenar por importancia absoluta
                resultados_lasso = resultados_lasso.sort_values('Importancia_Abs', ascending=False)
                
                # 🔍 IDENTIFICAR TOP Y BOTTOM FEATURES (SOLO ENTRE LAS ELIMINABLES)
                print(f"\n🔍 IDENTIFICANDO FEATURES A MANTENER Y ELIMINAR:")
                
                # Separar variables protegidas de las eliminables
                variables_protegidas_resultado = resultados_lasso[
                    resultados_lasso['Variable'].isin(variables_protegidas_disponibles)
                ]
                variables_eliminables_resultado = resultados_lasso[
                    resultados_lasso['Variable'].isin(variables_lasso_disponibles)
                ]
                
                # Calcular número de features eliminables a mantener
                num_features_eliminables = len(variables_lasso_disponibles)
                num_features_eliminables_mantener = max(1, int(num_features_eliminables * porcentaje_mantener / 100))
                num_features_eliminar = num_features_eliminables - num_features_eliminables_mantener
                
                print(f"   Variables PROTEGIDAS (siempre se mantienen): {len(variables_protegidas_disponibles)}")
                print(f"   Variables ELIMINABLES analizadas: {num_features_eliminables}")
                print(f"   Variables eliminables a MANTENER (top {porcentaje_mantener}%): {num_features_eliminables_mantener}")
                print(f"   Variables eliminables a ELIMINAR (bottom {100-porcentaje_mantener}%): {num_features_eliminar}")
                
                # Seleccionar features eliminables
                top_features_eliminables = variables_eliminables_resultado.head(num_features_eliminables_mantener)['Variable'].tolist()
                bottom_features_eliminables = variables_eliminables_resultado.iloc[num_features_eliminables_mantener:]['Variable'].tolist()
                
                # Features finales que se mantienen
                top_features = variables_protegidas_disponibles + top_features_eliminables
                bottom_features = bottom_features_eliminables  # Solo las eliminables van al bottom
                
                print(f"\n🔒 VARIABLES PROTEGIDAS (NUNCA SE ELIMINAN):")
                for i, feature in enumerate(variables_protegidas_disponibles, 1):
                    importancia = resultados_lasso[resultados_lasso['Variable'] == feature]['Importancia_Abs'].iloc[0]
                    print(f"   {i:2d}. {feature} (importancia: {importancia:.4f}) [PROTEGIDA]")
                
                print(f"\n🏆 TOP {porcentaje_mantener}% FEATURES ELIMINABLES (SE MANTIENEN):")
                for i, feature in enumerate(top_features_eliminables[:10], 1):  # Mostrar solo top 10
                    importancia = resultados_lasso[resultados_lasso['Variable'] == feature]['Importancia_Abs'].iloc[0]
                    print(f"   {i:2d}. {feature} (importancia: {importancia:.4f})")
                if len(top_features_eliminables) > 10:
                    print(f"   ... y {len(top_features_eliminables) - 10} más")
                
                print(f"\n🗑️ BOTTOM {100-porcentaje_mantener}% FEATURES ELIMINABLES (SE ELIMINAN):")
                if len(bottom_features) > 0:
                    for i, feature in enumerate(bottom_features[:10], 1):  # Mostrar solo primeras 10
                        importancia = resultados_lasso[resultados_lasso['Variable'] == feature]['Importancia_Abs'].iloc[0]
                        print(f"   {i:2d}. {feature} (importancia: {importancia:.4f})")
                    if len(bottom_features) > 10:
                        print(f"   ... y {len(bottom_features) - 10} más")
                else:
                    print("   (Ninguna)")
                
                # 🗑️ CREAR ARCHIVO FILTRADO
                nombre_posicion = archivo.replace('07_db_', '').replace('.csv', '').capitalize()
                
                print(f"\n🗑️ CREANDO ARCHIVO FILTRADO (ELIMINANDO BOTTOM {100-porcentaje_mantener}%):")
                
                # Columnas finales: variables no analizadas + protegidas + top eliminables + variable objetivo
                columnas_mantener = variables_no_lasso + top_features + [variable_objetivo]
                
                # Crear DataFrame filtrado
                df_filtrado = df[columnas_mantener].copy()
                
                # Guardar archivo filtrado
                output_path = f"DB_separadas/08_db_{nombre_posicion.lower()}_filtered_top{porcentaje_mantener}pct.csv"
                df_filtrado.to_csv(output_path, index=False)
                
                print(f"   ✅ Archivo filtrado creado: {output_path}")
                print(f"   📊 Dimensiones originales: {df.shape}")
                print(f"   📊 Dimensiones filtradas: {df_filtrado.shape}")
                print(f"   🔢 Columnas eliminadas: {df.shape[1] - df_filtrado.shape[1]}")
                print(f"   🔢 Columnas mantenidas: {df_filtrado.shape[1]}")
                
                # Desglose detallado
                print(f"\n📋 DESGLOSE DE COLUMNAS:")
                print(f"   Variables no analizadas (mantenidas): {len(variables_no_lasso)}")
                print(f"   Variables PROTEGIDAS (mantenidas): {len(variables_protegidas_disponibles)}")
                print(f"   Top {porcentaje_mantener}% features eliminables (mantenidas): {len(top_features_eliminables)}")
                print(f"   Bottom {100-porcentaje_mantener}% features eliminables (eliminadas): {len(bottom_features)}")
                print(f"   Variable objetivo (mantenida): 1")
                print(f"   Total mantenidas: {len(columnas_mantener)}")
                
                # Guardar resultados
                resultados_globales[archivo] = {
                    'resultados_lasso': resultados_lasso,
                    'top_features': top_features,
                    'bottom_features': bottom_features,
                    'variables_protegidas': variables_protegidas_disponibles,
                    'nombre_posicion': nombre_posicion
                }
                archivos_filtrados[archivo] = output_path
                
                # 💾 GUARDAR ANÁLISIS
                print(f"\n💾 GUARDANDO ANÁLISIS:")
                
                # Guardar ranking de importancia
                output_ranking = f"lasso/08_ranking_importancia_{nombre_posicion.lower()}_top{porcentaje_mantener}pct.csv"
                resultados_lasso.to_csv(output_ranking, index=False)
                print(f"   Ranking de importancia: {output_ranking}")
                
                # Guardar lista de features eliminadas
                if len(bottom_features) > 0:
                    df_eliminadas = pd.DataFrame({
                        'Feature_Eliminada': bottom_features,
                        'Importancia': [resultados_lasso[resultados_lasso['Variable'] == f]['Importancia_Abs'].iloc[0] 
                                       for f in bottom_features]
                    })
                    output_eliminadas = f"lasso/08_features_eliminadas_{nombre_posicion.lower()}_bottom{100-porcentaje_mantener}pct.csv"
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
                
            except Exception as e:
                print(f"❌ Error en LASSO: {str(e)}")
                continue
                
        except Exception as e:
            print(f"❌ Error procesando {archivo}: {str(e)}")
            continue
    
    # Resumen final
    print("\n" + "="*60)
    print("📋 RESUMEN FINAL")
    print("="*60)
    
    if resultados_globales:
        print(f"Archivos procesados exitosamente: {len(resultados_globales)}")
        print(f"Porcentaje de features eliminables mantenidas: {porcentaje_mantener}%")
        print(f"Porcentaje de features eliminables eliminadas: {100-porcentaje_mantener}%")
        print(f"🔒 Variables PROTEGIDAS: {', '.join(variables_protegidas)} (NUNCA se eliminan)")
        print(f"📁 Archivos filtrados guardados en: DB_separadas/")
        print(f"📁 Análisis guardados en: lasso/")
        
        print(f"\n📁 ARCHIVOS FILTRADOS CREADOS:")
        for archivo_original, archivo_filtrado in archivos_filtrados.items():
            print(f"   {archivo_original} → {archivo_filtrado}")
        
        # Análisis agregado
        print(f"\n🔍 ANÁLISIS AGREGADO:")
        
        # Variables protegidas disponibles en todos los archivos
        todas_variables_protegidas = []
        for archivo, datos in resultados_globales.items():
            variables_prot = datos.get('variables_protegidas', [])
            todas_variables_protegidas.extend(variables_prot)
        
        from collections import Counter
        contador_protegidas = Counter(todas_variables_protegidas)
        
        print(f"\n🔒 VARIABLES PROTEGIDAS (PRESENTES EN ARCHIVOS):")
        for var, freq in contador_protegidas.most_common():
            print(f"   {var}: presente en {freq} archivo(s) - NUNCA ELIMINADA")
        
        # Variables que aparecen frecuentemente como importantes (excluyendo protegidas)
        todas_variables_top = []
        todas_variables_bottom = []
        
        for archivo, datos in resultados_globales.items():
            # Filtrar solo las variables no protegidas para el análisis
            variables_prot = datos.get('variables_protegidas', [])
            top_vars = [v for v in datos['top_features'][:5] if v not in variables_prot]  # Top 5 eliminables
            bottom_vars = datos['bottom_features'][:5]  # Bottom 5 de cada archivo
            todas_variables_top.extend(top_vars)
            todas_variables_bottom.extend(bottom_vars)
        
        # Variables más frecuentes en top (eliminables)
        contador_top = Counter(todas_variables_top)
        vars_frecuentes_top = contador_top.most_common(10)
        
        print(f"\n🏆 Variables eliminables que más aparecen en TOP 5 (mantenidas):")
        for var, freq in vars_frecuentes_top:
            print(f"   {var}: {freq} veces")
        
        # Variables más frecuentes en bottom
        contador_bottom = Counter(todas_variables_bottom)
        vars_frecuentes_bottom = contador_bottom.most_common(10)
        
        print(f"\n🗑️ Variables que más aparecen en BOTTOM 5 (eliminadas):")
        for var, freq in vars_frecuentes_bottom:
            print(f"   {var}: {freq} veces")
        
        # Guardar resumen
        print(f"\n💾 GUARDANDO RESUMEN FINAL:")
        
        # Resumen de features mantenidas por posición
        resumen_mantenidas = pd.DataFrame()
        for archivo, datos in resultados_globales.items():
            posicion = datos['nombre_posicion']
            features = datos['top_features']
            variables_prot = datos.get('variables_protegidas', [])
            temp_df = pd.DataFrame({
                'Posicion': [posicion] * len(features),
                'Feature_Mantenida': features,
                'Es_Protegida': [f in variables_prot for f in features],
                'Rank': range(1, len(features) + 1)
            })
            resumen_mantenidas = pd.concat([resumen_mantenidas, temp_df], ignore_index=True)
        
        resumen_mantenidas.to_csv(f'lasso/08_resumen_features_mantenidas_top{porcentaje_mantener}pct.csv', index=False)
        print(f"   Resumen features mantenidas: lasso/08_resumen_features_mantenidas_top{porcentaje_mantener}pct.csv")
        
        # Resumen de features eliminadas por posición
        resumen_eliminadas = pd.DataFrame()
        for archivo, datos in resultados_globales.items():
            posicion = datos['nombre_posicion']
            features = datos['bottom_features']
            if len(features) > 0:
                temp_df = pd.DataFrame({
                    'Posicion': [posicion] * len(features),
                    'Feature_Eliminada': features,
                    'Rank_Eliminacion': range(1, len(features) + 1)
                })
                resumen_eliminadas = pd.concat([resumen_eliminadas, temp_df], ignore_index=True)
        
        if not resumen_eliminadas.empty:
            resumen_eliminadas.to_csv(f'lasso/08_resumen_features_eliminadas_bottom{100-porcentaje_mantener}pct.csv', index=False)
            print(f"   Resumen features eliminadas: lasso/08_resumen_features_eliminadas_bottom{100-porcentaje_mantener}pct.csv")
        
        # Resumen de variables protegidas
        resumen_protegidas = pd.DataFrame()
        for archivo, datos in resultados_globales.items():
            posicion = datos['nombre_posicion']
            variables_prot = datos.get('variables_protegidas', [])
            if len(variables_prot) > 0:
                temp_df = pd.DataFrame({
                    'Posicion': [posicion] * len(variables_prot),
                    'Variable_Protegida': variables_prot,
                    'Estado': ['PROTEGIDA - NUNCA ELIMINADA'] * len(variables_prot)
                })
                resumen_protegidas = pd.concat([resumen_protegidas, temp_df], ignore_index=True)
        
        if not resumen_protegidas.empty:
            resumen_protegidas.to_csv(f'lasso/08_variables_protegidas_resumen.csv', index=False)
            print(f"   Resumen variables protegidas: lasso/08_variables_protegidas_resumen.csv")
        
        print(f"\n🎯 PROCESO COMPLETADO:")
        print(f"   ✅ {len(resultados_globales)} archivos procesados")
        print(f"   🔒 Variables PROTEGIDAS: {', '.join(variables_protegidas)} (NUNCA eliminadas)")
        print(f"   ✅ Bottom {100-porcentaje_mantener}% de features ELIMINABLES eliminadas")
        print(f"   ✅ Archivos filtrados guardados con prefijo '08_'")
        print(f"   ✅ Análisis detallados guardados en carpeta 'lasso/'")
    
    else:
        print("❌ No se pudo procesar ningún archivo")

# Ejecutar análisis
if __name__ == "__main__":
    print("🚀 INICIANDO ANÁLISIS LASSO - ELIMINANDO BOTTOM 60% DE FEATURES")
    print("="*60)
    analizar_importancia_lasso(porcentaje_mantener=40)