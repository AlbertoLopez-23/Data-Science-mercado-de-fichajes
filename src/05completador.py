import pandas as pd
import numpy as np

def procesar_csv_con_nulls(archivo_entrada, archivo_salida):
    """
    Lee un archivo CSV, reemplaza campos vacíos con NULL y guarda el resultado.
    
    Args:
        archivo_entrada (str): Ruta del archivo CSV de entrada
        archivo_salida (str): Ruta del archivo CSV de salida
    """
    
    try:
        # Leer el archivo CSV
        print(f"Leyendo archivo: {archivo_entrada}")
        df = pd.read_csv(archivo_entrada)
        
        # Mostrar número de filas del archivo de entrada
        filas_entrada = len(df)
        print(f"Número de filas en el archivo de entrada: {filas_entrada}")
        
        # Reemplazar valores vacíos con 'NULL'
        # Esto incluye: NaN, strings vacíos, espacios en blanco
        df = df.replace('', 'NULL')  # Strings vacíos
        df = df.replace(r'^\s*$', 'NULL', regex=True)  # Solo espacios en blanco
        df = df.fillna('NULL')  # Valores NaN
        
        # Guardar el archivo procesado
        df.to_csv(archivo_salida, index=False)
        print(f"Archivo guardado como: {archivo_salida}")
        
        # Mostrar número de filas del archivo de salida
        filas_salida = len(df)
        print(f"Número de filas en el archivo de salida: {filas_salida}")
        
        # Mostrar información adicional
        print(f"\nResumen del procesamiento:")
        print(f"- Columnas procesadas: {len(df.columns)}")
        print(f"- Total de celdas: {df.shape[0] * df.shape[1]}")
        
        # Contar cuántas celdas fueron reemplazadas por NULL
        null_count = (df == 'NULL').sum().sum()
        print(f"- Celdas reemplazadas con NULL: {null_count}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_entrada}")
        return None
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        return None

# Ejecutar el procesamiento
if __name__ == "__main__":
    archivo_entrada = "05_db_normalizado.csv"
    archivo_salida = "06_db_completo.csv"
    
    resultado = procesar_csv_con_nulls(archivo_entrada, archivo_salida)
    
    if resultado is not None:
        print("\n¡Procesamiento completado exitosamente!")
    else:
        print("\nEl procesamiento falló. Revisa los errores anteriores.")