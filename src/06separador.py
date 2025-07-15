import pandas as pd
import os

def separar_archivo_por_posicion():
    # Nombre del archivo de entrada
    archivo_entrada = '06_db_completo.csv'
    
    # Diccionario para mapear números a nombres de posiciones
    posiciones = {
        0: "Centrocampista",
        1: "Defensa", 
        2: "Delantero",
        3: "Portero"
    }
    
    try:
        # Leer el archivo CSV tratando los espacios en blanco como NaN
        print(f"Leyendo archivo: {archivo_entrada}")
        df = pd.read_csv(archivo_entrada, na_values=[' ', '', 'null', 'NULL', 'nan', 'NaN', 'NA'])
        
        # Mostrar información del archivo de entrada
        filas_entrada, columnas_entrada = df.shape
        print(f"\n--- ARCHIVO DE ENTRADA ---")
        print(f"Número de filas: {filas_entrada}")
        print(f"Número de columnas: {columnas_entrada}")
        print(f"Columnas disponibles: {list(df.columns)}")
        
        # Verificar que existe la columna 'Posición principal'
        if 'Posición principal' not in df.columns:
            print("Error: No se encuentra la columna 'Posición principal'")
            return
        
        # Mostrar distribución de posiciones
        print(f"\n--- DISTRIBUCIÓN DE POSICIONES ---")
        distribucion = df['Posición principal'].value_counts().sort_index()
        for pos_num, cantidad in distribucion.items():
            nombre_pos = posiciones.get(pos_num, f"Posición_{pos_num}")
            print(f"{nombre_pos} ({pos_num}): {cantidad} jugadores")
        
        # Separar por posición y guardar archivos
        print(f"\n--- CREANDO ARCHIVOS SEPARADOS ---")
        
        for posicion_num, nombre_posicion in posiciones.items():
            # Filtrar jugadores de esta posición
            df_posicion = df[df['Posición principal'] == posicion_num]
            
            if len(df_posicion) > 0:
                # Crear nombre de archivo
                nombre_archivo = f"07_db_{nombre_posicion.lower()}.csv"
                
                # Guardar archivo manteniendo los valores nulos
                df_posicion.to_csv(nombre_archivo, index=False, na_rep='NULL')
                
                # Mostrar información del archivo creado
                filas_salida, columnas_salida = df_posicion.shape
                print(f"\n{nombre_posicion}:")
                print(f"  - Archivo: {nombre_archivo}")
                print(f"  - Filas: {filas_salida}")
                print(f"  - Columnas: {columnas_salida}")
            else:
                print(f"\n{nombre_posicion}: No hay jugadores de esta posición")
        
        print(f"\n--- PROCESO COMPLETADO ---")
        print("Archivos creados exitosamente!")
        
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo '{archivo_entrada}'")
        print("Asegúrate de que el archivo esté en el mismo directorio que este script")
    
    except Exception as e:
        print(f"Error inesperado: {str(e)}")

# Ejecutar la función
if __name__ == "__main__":
    separar_archivo_por_posicion()