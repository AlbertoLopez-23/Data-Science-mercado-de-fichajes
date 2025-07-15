import pandas as pd

def reordenar_columnas_csv():
    """
    Lee el archivo CSV '02_df_columnas_eliminadas.csv', reordena las columnas 
    según el orden especificado y guarda el resultado en '03_db_columnas_ordenadas.csv'
    """
    
    # Leer el archivo CSV original
    try:
        df = pd.read_csv('02_df_columnas_eliminadas.csv')
        filas_entrada = df.shape[0]
        columnas_entrada = df.shape[1]
        print(f"=== ARCHIVO DE ENTRADA ===")
        print(f"Archivo leído exitosamente: '02_df_columnas_eliminadas.csv'")
        print(f"Número de filas: {filas_entrada}")
        print(f"Número de columnas: {columnas_entrada}")
        print(f"Dimensiones totales: {df.shape}")
    except FileNotFoundError:
        print("Error: No se encontró el archivo '02_df_columnas_eliminadas.csv'")
        return
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return
    
    # Orden deseado de columnas
    orden_columnas = [
        'Nombre completo',
        'Lugar de nacimiento (país)',
        'Nacionalidad',
        'Posición principal',
        'Posición específica',
        'Posición secundaria',
        'Club actual',
        'Proveedor',
        'Fin de contrato',
        'Valor de mercado actual (numérico)',
        'Fecha de fichaje',
        'comprado_por',
        'Pie bueno',
        'understat_matches',
        'understat_minutes',
        'understat_goals',
        'understat_xg',
        'understat_assists',
        'understat_xa',
        'understat_shots',
        'understat_key_passes',
        'understat_yellow_cards',
        'understat_red_cards',
        'overallrating',
        'potential',
        'crossing',
        'finishing',
        'headingaccuracy',
        'shortpassing',
        'volleys',
        'dribbling',
        'curve',
        'fk_accuracy',
        'longpassing',
        'ballcontrol',
        'acceleration',
        'sprintspeed',
        'agility',
        'reactions',
        'balance',
        'shotpower',
        'jumping',
        'stamina',
        'strength',
        'longshots',
        'aggression',
        'interceptions',
        'positioning',
        'vision',
        'penalties',
        'composure',
        'defensiveawareness',
        'standingtackle',
        'slidingtackle',
        'gk_diving',
        'gk_handling',
        'gk_kicking',
        'gk_positioning',
        'gk_reflexes'
    ]
    
    # Verificar qué columnas existen en el DataFrame
    columnas_existentes = []
    columnas_faltantes = []
    
    for col in orden_columnas:
        if col in df.columns:
            columnas_existentes.append(col)
        else:
            columnas_faltantes.append(col)
    
    # Agregar columnas que están en el DataFrame pero no en la lista de orden
    columnas_extra = [col for col in df.columns if col not in orden_columnas]
    
    if columnas_faltantes:
        print(f"Advertencia: Las siguientes columnas no se encontraron en el archivo:")
        for col in columnas_faltantes:
            print(f"  - {col}")
    
    if columnas_extra:
        print(f"Advertencia: Las siguientes columnas están en el archivo pero no en el orden especificado:")
        for col in columnas_extra:
            print(f"  - {col}")
        print("Estas columnas se agregarán al final.")
    
    # Crear el nuevo orden final (columnas existentes + columnas extra)
    orden_final = columnas_existentes + columnas_extra
    
    # Reordenar el DataFrame
    df_reordenado = df[orden_final]
    
    # Guardar el archivo CSV reordenado
    try:
        df_reordenado.to_csv('03_db_columnas_ordenadas.csv', index=False)
        filas_salida = df_reordenado.shape[0]
        columnas_salida = df_reordenado.shape[1]
        
        print(f"\n=== ARCHIVO DE SALIDA ===")
        print(f"Archivo guardado exitosamente como '03_db_columnas_ordenadas.csv'")
        print(f"Número de filas: {filas_salida}")
        print(f"Número de columnas: {columnas_salida}")
        print(f"Dimensiones totales: {df_reordenado.shape}")
        
        print(f"\n=== RESUMEN COMPARATIVO ===")
        print(f"Archivo entrada:  {filas_entrada} filas × {columnas_entrada} columnas")
        print(f"Archivo salida:   {filas_salida} filas × {columnas_salida} columnas")
        
        if filas_entrada == filas_salida and columnas_entrada == columnas_salida:
            print("✓ Las dimensiones se mantuvieron iguales (solo se reordenaron las columnas)")
        elif filas_entrada == filas_salida:
            print(f"✓ El número de filas se mantuvo igual")
            if columnas_entrada != columnas_salida:
                print(f"⚠ El número de columnas cambió de {columnas_entrada} a {columnas_salida}")
        else:
            print("⚠ Hubo cambios en las dimensiones del archivo")
            
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

# Ejecutar la función
if __name__ == "__main__":
    reordenar_columnas_csv()