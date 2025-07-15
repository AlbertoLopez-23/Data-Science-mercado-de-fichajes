import pandas as pd

# Especifica la ruta de tu archivo CSV
archivo_csv = "01_df_filtrado_final.csv"  # Cambia esto por la ruta real de tu archivo

# Lista de columnas a eliminar
columnas_a_eliminar = [
    'URL',
    'match_similarity',
    'match_confidence',
    'normalized_complete_name',
    'normalized_fbref_name',
    'fifa_edition',
    'update',
    'nombre_id',
    'ID',
    'Valor de mercado actual'
]

try:
    # Leer el archivo CSV
    print(f"Leyendo el archivo: {archivo_csv}")
    df = pd.read_csv(archivo_csv)
    
    # Mostrar información inicial
    print(f"Dimensiones originales: {df.shape}")
    print(f"Número de filas: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    print(f"Columnas originales: {list(df.columns)}")
    
    # Verificar qué columnas existen realmente en el DataFrame
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    columnas_no_encontradas = [col for col in columnas_a_eliminar if col not in df.columns]
    
    if columnas_no_encontradas:
        print(f"\nAdvertencia: Las siguientes columnas no se encontraron en el archivo:")
        for col in columnas_no_encontradas:
            print(f"  - {col}")
    
    if columnas_existentes:
        # Eliminar las columnas que sí existen
        df_limpio = df.drop(columns=columnas_existentes)
        print(f"\nColumnas eliminadas: {columnas_existentes}")
        print(f"Dimensiones finales: {df_limpio.shape}")
        print(f"Número de filas final: {df_limpio.shape[0]}")
        print(f"Número de columnas final: {df_limpio.shape[1]}")
        print(f"Columnas restantes: {list(df_limpio.columns)}")
        
        # Guardar el archivo modificado
        archivo_salida = '02_df_columnas_eliminadas.csv'
        df_limpio.to_csv(archivo_salida, index=False)
        print(f"\nArchivo guardado como: {archivo_salida}")
        
        # Opcional: Mostrar las primeras filas del resultado
        print("\nPrimeras 5 filas del archivo limpio:")
        print(df_limpio.head())
        
    else:
        print("\nNo se encontraron columnas para eliminar.")
        
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{archivo_csv}'")
    print("Asegúrate de que la ruta del archivo sea correcta.")
    
except pd.errors.EmptyDataError:
    print("Error: El archivo CSV está vacío.")
    
except pd.errors.ParserError as e:
    print(f"Error al leer el archivo CSV: {e}")
    
except Exception as e:
    print(f"Error inesperado: {e}")