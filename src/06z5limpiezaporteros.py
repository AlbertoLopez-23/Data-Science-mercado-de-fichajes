import pandas as pd
import os

def procesar_csv_porteros():
    # Definir rutas de archivos
    archivo_entrada = 'DB_viejas/06.5_db_portero.csv'
    archivo_salida = '07_db_portero.csv'
    
    # Columnas a eliminar
    columnas_eliminar = [
        'Posición específica',
        'Posición secundaria', 
        'understat_goals',
        'understat_xg',
        'understat_assists',
        'understat_xa',
        'understat_shots',
        'understat_key_passes',
        'understat_yellow_cards',
        'understat_red_cards'
    ]
    
    try:
        # Verificar que el archivo de entrada existe
        if not os.path.exists(archivo_entrada):
            print(f"Error: No se encontró el archivo {archivo_entrada}")
            return
        
        # Leer el archivo CSV tratando los espacios en blanco como NaN
        print(f"Leyendo archivo: {archivo_entrada}")
        df = pd.read_csv(archivo_entrada, na_values=[' ', '', 'null', 'NULL', 'nan', 'NaN', 'NA'])
        
        print(f"Archivo original: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Verificar qué columnas existen en el DataFrame
        columnas_existentes = df.columns.tolist()
        columnas_a_eliminar_existentes = []
        columnas_no_encontradas = []
        
        for col in columnas_eliminar:
            if col in columnas_existentes:
                columnas_a_eliminar_existentes.append(col)
            else:
                columnas_no_encontradas.append(col)
        
        # Mostrar información sobre las columnas
        if columnas_a_eliminar_existentes:
            print(f"\nColumnas que se eliminarán: {columnas_a_eliminar_existentes}")
        
        if columnas_no_encontradas:
            print(f"Columnas no encontradas en el archivo: {columnas_no_encontradas}")
        
        # Eliminar las columnas que existen
        if columnas_a_eliminar_existentes:
            df_limpio = df.drop(columns=columnas_a_eliminar_existentes)
        else:
            df_limpio = df.copy()
            print("No se eliminó ninguna columna porque no se encontraron las columnas especificadas")
        
        print(f"Archivo procesado: {df_limpio.shape[0]} filas, {df_limpio.shape[1]} columnas")
        
        # Guardar el archivo limpio manteniendo los valores nulos
        df_limpio.to_csv(archivo_salida, index=False, na_rep='NULL')
        print(f"\nArchivo guardado exitosamente como: {archivo_salida}")
        
        # Mostrar las primeras columnas del archivo resultante
        print(f"\nPrimeras columnas del archivo resultante:")
        print(df_limpio.columns.tolist()[:10])  # Mostrar las primeras 10 columnas
        
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")

# Ejecutar la función
if __name__ == "__main__":
    procesar_csv_porteros()