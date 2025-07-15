(.venv) alberto@alberto-torre:~/Master/Codigos/Codigo-TFM-2$ python 07Lasso.py
🚀 INICIANDO ANÁLISIS LASSO PARA IMPORTANCIA DE VARIABLES
============================================================
Encontrados 3 archivos CSV para analizar
Archivos a procesar: ['07_db_delantero.csv', '07_db_defensa.csv', '07_db_centrocampista.csv']
============================================================

📊 ANÁLISIS DE: 07_db_delantero.csv
--------------------------------------------------
Dimensiones del dataset: (447, 59)
Variables disponibles: 44
Datos después de limpieza: 447 muestras
Datos de entrenamiento: 357 muestras
Datos de prueba: 90 muestras
Entrenando modelo con validación cruzada...

📈 RESULTADOS DEL MODELO:
Alpha óptimo: 100.000000
R² Entrenamiento: 0.6757
R² Prueba: 0.4317
RMSE Entrenamiento: $13,284,215
RMSE Prueba: $16,795,139

Variables importantes (coef ≠ 0): 44
Variables eliminadas por LASSO: 0

💾 Predicciones guardadas en: predicciones_delantero_lasso.csv

📊 ESTADÍSTICAS DE ERROR:
Error porcentual promedio: inf%
Error porcentual mediano: 112.0%
Predicciones dentro del 20% del valor real: 15.6%
Predicciones dentro del 50% del valor real: 36.7%
📊 Gráfico guardado: importancia_variables_delantero.png

📊 ANÁLISIS DE: 07_db_defensa.csv
--------------------------------------------------
Dimensiones del dataset: (671, 59)
Variables disponibles: 44
Datos después de limpieza: 671 muestras
Datos de entrenamiento: 536 muestras
Datos de prueba: 135 muestras
Entrenando modelo con validación cruzada...

📈 RESULTADOS DEL MODELO:
Alpha óptimo: 100.000000
R² Entrenamiento: 0.6128
R² Prueba: 0.6286
RMSE Entrenamiento: $8,535,184
RMSE Prueba: $8,114,699

Variables importantes (coef ≠ 0): 44
Variables eliminadas por LASSO: 0

💾 Predicciones guardadas en: predicciones_defensa_lasso.csv

📊 ESTADÍSTICAS DE ERROR:
Error porcentual promedio: inf%
Error porcentual mediano: 50.9%
Predicciones dentro del 20% del valor real: 18.5%
Predicciones dentro del 50% del valor real: 48.1%
📊 Gráfico guardado: importancia_variables_defensa.png

📊 ANÁLISIS DE: 07_db_centrocampista.csv
--------------------------------------------------
Dimensiones del dataset: (529, 59)
Variables disponibles: 44
Datos después de limpieza: 529 muestras
Datos de entrenamiento: 423 muestras
Datos de prueba: 106 muestras
Entrenando modelo con validación cruzada...

📈 RESULTADOS DEL MODELO:
Alpha óptimo: 100.000000
R² Entrenamiento: 0.6866
R² Prueba: 0.3427
RMSE Entrenamiento: $11,784,719
RMSE Prueba: $12,756,650

Variables importantes (coef ≠ 0): 44
Variables eliminadas por LASSO: 0

💾 Predicciones guardadas en: predicciones_centrocampista_lasso.csv

📊 ESTADÍSTICAS DE ERROR:
Error porcentual promedio: 347.5%
Error porcentual mediano: 95.4%
Predicciones dentro del 20% del valor real: 9.4%
Predicciones dentro del 50% del valor real: 35.8%
📊 Gráfico guardado: importancia_variables_centrocampista.png

============================================================
📋 RESUMEN FINAL
============================================================
Archivos procesados exitosamente: 3
💾 Guardado: importancia_07_db_delantero.csv
💾 Guardado: importancia_07_db_defensa.csv
💾 Guardado: importancia_07_db_centrocampista.csv

🔍 ANÁLISIS AGREGADO:

🏆 Variables que más aparecen en TOP 10:
  jumping: 3 veces
  potential: 3 veces
  strength: 3 veces
  headingaccuracy: 3 veces
  overallrating: 2 veces
  acceleration: 2 veces
  standingtackle: 1 veces
  slidingtackle: 1 veces
  understat_xg: 1 veces
  agility: 1 veces

📊 RANKING PROMEDIO DE IMPORTANCIA:
Variable | Importancia Promedio | Archivos
--------------------------------------------------
jumping                        | 7727227.5730 |        3
potential                      | 6844611.6760 |        3
overallrating                  | 6769042.6398 |        3
strength                       | 4853460.2542 |        3
standingtackle                 | 3879244.7556 |        3
headingaccuracy                | 3685154.3394 |        3
acceleration                   | 3408393.4093 |        3
understat_matches              | 2984921.3835 |        3
slidingtackle                  | 2939146.4198 |        3
understat_xg                   | 2790648.1316 |        3
ballcontrol                    | 2767335.9847 |        3
understat_goals                | 2697550.8939 |        3
vision                         | 2655585.9126 |        3
understat_minutes              | 2434532.6458 |        3
crossing                       | 2203332.2148 |        3
(.venv) alberto@alberto-torre:~/Master/Codigos/Codigo-TFM-2$ python 07zLasso.py
🚀 INICIANDO ANÁLISIS LASSO PARA PORTEROS
============================================================
🥅 ANÁLISIS LASSO PARA PORTEROS
============================================================
📂 Cargando datos desde: DB_separadas/07_db_portero.csv
   Dimensiones originales: (228, 49)
   Variables disponibles: 39/39

📊 INFORMACIÓN DEL DATASET:
   Total de porteros: 228
   Valor de mercado - Min: $0
   Valor de mercado - Max: $40,000,000
   Valor de mercado - Promedio: $5,251,864

🧹 LIMPIEZA DE DATOS:
   Después de eliminar valores faltantes en objetivo: 228 muestras
   Valores faltantes por variable:
     Pie bueno: 3 (1.3%)
     understat_matches: 74 (32.5%)
     understat_minutes: 74 (32.5%)
   Dataset final: 228 porteros, 39 variables

🎯 EJECUTANDO ANÁLISIS LASSO:
   Datos de entrenamiento: 182 muestras
   Datos de prueba: 46 muestras
   Entrenando modelo con validación cruzada...

📈 RESULTADOS DEL MODELO:
   Alpha óptimo: 100.000000
   R² Entrenamiento: 0.6883
   R² Prueba: 0.1174
   RMSE Entrenamiento: $4,601,041
   RMSE Prueba: $6,440,998

📊 ANÁLISIS DE IMPORTANCIA DE VARIABLES:
   Variables importantes (coef ≠ 0): 39
   Variables eliminadas por LASSO: 0

🏆 TOP 15 VARIABLES MÁS IMPORTANTES:
----------------------------------------------------------------------
Variable                  Coeficiente     Importancia    
----------------------------------------------------------------------
 1. overallrating          +30510264.4914 30510264.4914
 2. shotpower              -23210245.8887 23210245.8887
 3. gk_kicking             +22458040.2963 22458040.2963
 4. gk_reflexes            -9353134.2885  9353134.2885
 5. gk_diving              -9066080.0206  9066080.0206
 6. gk_handling            -5786665.7350  5786665.7350
 7. gk_positioning         -5125179.5163  5125179.5163
 8. reactions              -4753525.8359  4753525.8359
 9. longshots              -3603269.9171  3603269.9171
10. ballcontrol            +3272712.8110  3272712.8110
11. volleys                +3214259.4196  3214259.4196
12. positioning            -3112980.2582  3112980.2582
13. finishing              -2964359.1341  2964359.1341
14. potential              +2376190.7462  2376190.7462
15. crossing               -2113935.3356  2113935.3356

🥅 HABILIDADES ESPECÍFICAS DE PORTERO:
--------------------------------------------------
   gk_diving            -9066080.0206 (Rank: 35)
   gk_handling          -5786665.7350 (Rank: 36)
   gk_kicking           +22458040.2963 (Rank: 37)
   gk_positioning       -5125179.5163 (Rank: 38)
   gk_reflexes          -9353134.2885 (Rank: 39)

💾 GUARDANDO RESULTADOS:
   Importancia de variables: importancia_porteros_lasso.csv
   Predicciones del modelo: predicciones_porteros_lasso.csv

📊 ESTADÍSTICAS DE ERROR:
   Error porcentual promedio: 767.5%
   Error porcentual mediano: 112.0%
   Predicciones dentro del 20% del valor real: 17.4%
   Predicciones dentro del 50% del valor real: 37.0%

🎯 RECOMENDACIONES:
   ❌ Modelo tiene bajo poder predictivo (R² = 0.117)
   🔑 Las 3 variables más importantes son: overallrating, shotpower, gk_kicking

📊 Creando gráficos de importancia...
📊 Gráfico guardado: importancia_variables_porteros.png

✅ ANÁLISIS COMPLETADO