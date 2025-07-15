import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Leer el archivo CSV
df = pd.read_csv('04_db_codificado.csv')

# Mostrar información del archivo de entrada
print("ARCHIVO DE ENTRADA:")
print(f"Número de filas: {df.shape[0]}")
print(f"Número de columnas: {df.shape[1]}")
print("-" * 50)

# Lista de variables a normalizar
variables_a_normalizar = [
    'comprado_por', 'understat_matches', 'understat_minutes', 'understat_goals', 
    'understat_xg', 'understat_assists', 'understat_xa', 'understat_shots', 
    'understat_key_passes', 'understat_yellow_cards', 'understat_red_cards', 
    'overallrating', 'potential', 'crossing', 'finishing', 'headingaccuracy', 
    'shortpassing', 'volleys', 'dribbling', 'curve', 'fk_accuracy', 
    'longpassing', 'ballcontrol', 'acceleration', 'sprintspeed', 'agility', 
    'reactions', 'balance', 'shotpower', 'jumping', 'stamina', 'strength', 
    'longshots', 'aggression', 'interceptions', 'positioning', 'vision', 
    'penalties', 'composure', 'defensiveawareness', 'standingtackle', 
    'slidingtackle', 'gk_diving', 'gk_handling', 'gk_kicking', 
    'gk_positioning', 'gk_reflexes'
]

# Verificar que las columnas existen en el DataFrame
columnas_existentes = [col for col in variables_a_normalizar if col in df.columns]
columnas_faltantes = [col for col in variables_a_normalizar if col not in df.columns]

if columnas_faltantes:
    print("ADVERTENCIA: Las siguientes columnas no se encontraron en el archivo:")
    print(columnas_faltantes)
    print("-" * 50)

# Crear una copia del DataFrame original
df_normalizado = df.copy()

# Inicializar el StandardScaler
scaler = StandardScaler()

# Aplicar StandardScaler solo a las columnas existentes
if columnas_existentes:
    # Manejar valores NaN antes de la normalización
    df_temp = df[columnas_existentes].copy()
    
    # Reemplazar valores infinitos con NaN
    df_temp = df_temp.replace([np.inf, -np.inf], np.nan)
    
    # Aplicar StandardScaler (ignora automáticamente los valores NaN)
    datos_normalizados = scaler.fit_transform(df_temp)
    
    # Reemplazar las columnas normalizadas en el DataFrame
    df_normalizado[columnas_existentes] = datos_normalizados
    
    print(f"Se normalizaron {len(columnas_existentes)} variables con StandardScaler")
else:
    print("No se encontraron columnas válidas para normalizar")

# Guardar el archivo normalizado
df_normalizado.to_csv('04_db_normalizado.csv', index=False)

# Mostrar información del archivo de salida
print("-" * 50)
print("ARCHIVO DE SALIDA:")
print(f"Número de filas: {df_normalizado.shape[0]}")
print(f"Número de columnas: {df_normalizado.shape[1]}")
print("-" * 50)
print("Archivo '04_db_normalizado.csv' guardado exitosamente")

# Mostrar estadísticas de las variables normalizadas (opcional)
print("-" * 50)
print("ESTADÍSTICAS DE LAS VARIABLES NORMALIZADAS:")
print("Media (debería ser ~0):")
print(df_normalizado[columnas_existentes].mean().round(6))
print("\nDesviación estándar (debería ser ~1):")
print(df_normalizado[columnas_existentes].std().round(6))