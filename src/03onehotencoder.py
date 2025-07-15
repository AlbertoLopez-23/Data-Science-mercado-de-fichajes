import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np

def codificar_variables_categoricas():
    """
    Codifica variables categóricas de la base de datos de fútbol
    """
    
    # Variables categóricas a codificar
    variables_categoricas = [
        'Nacionalidad',
        'Lugar de nacimiento (país)', 
        'Posición principal',
        'Posición específica',
        'Posición secundaria',
        'Club actual',
        'Proveedor', 
        'Pie bueno'
    ]
    
    try:
        # Leer el archivo CSV
        print("Leyendo archivo 03_db_columnas_ordenadas.csv...")
        df = pd.read_csv('03_db_columnas_ordenadas.csv')
        
        print(f"Datos de entrada:")
        print(f"- Número de filas: {df.shape[0]}")
        print(f"- Número de columnas: {df.shape[1]}")
        print(f"- Columnas encontradas: {list(df.columns)}")
        
        # Crear una copia para trabajar
        df_codificado = df.copy()
        
        # Diccionario para almacenar las codificaciones
        diccionario_codificaciones = {}
        
        # Procesar cada variable categórica
        for variable in variables_categoricas:
            if variable in df.columns:
                print(f"\nProcesando variable: {variable}")
                
                # Manejar valores NaN/nulos
                valores_unicos = df[variable].dropna().unique()
                print(f"  - Valores únicos encontrados: {len(valores_unicos)}")
                
                # Crear codificador
                le = LabelEncoder()
                
                # Ajustar el codificador solo con valores no nulos
                le.fit(valores_unicos)
                
                # Aplicar codificación, manteniendo NaN como NaN
                mask_no_nulos = df[variable].notna()
                df_codificado.loc[mask_no_nulos, variable] = le.transform(df.loc[mask_no_nulos, variable])
                
                # Guardar el mapeo en el diccionario
                mapeo = {str(valor): int(codigo) for codigo, valor in enumerate(le.classes_)}
                diccionario_codificaciones[variable] = {
                    'mapeo': mapeo,
                    'valores_originales': list(le.classes_),
                    'total_categorias': len(le.classes_)
                }
                
                print(f"  - Codificación completada: {len(le.classes_)} categorías")
                
            else:
                print(f"⚠️  Variable '{variable}' no encontrada en el dataset")
        
        # Mostrar información de la base de datos codificada
        print(f"\nDatos de salida:")
        print(f"- Número de filas: {df_codificado.shape[0]}")
        print(f"- Número de columnas: {df_codificado.shape[1]}")
        
        # Guardar el DataFrame codificado
        print("\nGuardando archivo codificado...")
        df_codificado.to_csv('04_db_codificado.csv', index=False)
        print("✅ Archivo '04_db_codificado.csv' guardado exitosamente")
        
        # Guardar el diccionario de codificaciones
        print("Guardando diccionario de codificaciones...")
        with open('diccionario_codificaciones.json', 'w', encoding='utf-8') as f:
            json.dump(diccionario_codificaciones, f, ensure_ascii=False, indent=2)
        print("✅ Archivo 'diccionario_codificaciones.json' guardado exitosamente")
        
        # Mostrar resumen de codificaciones
        print("\n" + "="*60)
        print("RESUMEN DE CODIFICACIONES")
        print("="*60)
        
        for variable, info in diccionario_codificaciones.items():
            print(f"\n{variable}:")
            print(f"  - Total de categorías: {info['total_categorias']}")
            print(f"  - Primeras 5 codificaciones:")
            mapeo_items = list(info['mapeo'].items())[:5]
            for valor_orig, codigo in mapeo_items:
                print(f"    '{valor_orig}' → {codigo}")
            if len(info['mapeo']) > 5:
                print(f"    ... y {len(info['mapeo']) - 5} más")
        
        # Verificar tipos de datos después de la codificación
        print(f"\n" + "="*60)
        print("VERIFICACIÓN DE TIPOS DE DATOS")
        print("="*60)
        
        for variable in variables_categoricas:
            if variable in df_codificado.columns:
                tipo_original = df[variable].dtype
                tipo_codificado = df_codificado[variable].dtype
                valores_nulos = df_codificado[variable].isna().sum()
                print(f"{variable}:")
                print(f"  - Tipo original: {tipo_original}")
                print(f"  - Tipo codificado: {tipo_codificado}")
                print(f"  - Valores nulos: {valores_nulos}")
        
        return df_codificado, diccionario_codificaciones
        
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo '03_db_columnas_ordenadas.csv'")
        print("   Asegúrate de que el archivo esté en el directorio actual")
        return None, None
        
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        return None, None

def mostrar_ejemplo_decodificacion(diccionario_codificaciones):
    """
    Muestra un ejemplo de cómo decodificar los valores
    """
    print("\n" + "="*60)
    print("EJEMPLO DE DECODIFICACIÓN")
    print("="*60)
    
    print("Para decodificar un valor, usa el diccionario inverso:")
    print("Ejemplo con 'Nacionalidad':")
    
    if 'Nacionalidad' in diccionario_codificaciones:
        mapeo = diccionario_codificaciones['Nacionalidad']['mapeo']
        # Crear mapeo inverso
        mapeo_inverso = {v: k for k, v in mapeo.items()}
        print(f"Código 0 = '{mapeo_inverso.get(0, 'No encontrado')}'")
        print(f"Código 1 = '{mapeo_inverso.get(1, 'No encontrado')}'")
        
        print("\nCódigo Python para decodificar:")
        print("mapeo_inverso = {v: k for k, v in diccionario_codificaciones['Nacionalidad']['mapeo'].items()}")
        print("valor_original = mapeo_inverso[codigo_numerico]")

if __name__ == "__main__":
    print("🏈 CODIFICADOR DE VARIABLES CATEGÓRICAS - BASE DE DATOS DE FÚTBOL")
    print("="*70)
    
    # Ejecutar la codificación
    df_resultado, dict_codificaciones = codificar_variables_categoricas()
    
    if df_resultado is not None:
        # Mostrar ejemplo de decodificación
        mostrar_ejemplo_decodificacion(dict_codificaciones)
        
        print("\n" + "="*70)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print("Archivos generados:")
        print("1. 04_db_codificado.csv - Base de datos con variables codificadas")
        print("2. diccionario_codificaciones.json - Mapeo de codificaciones")
    else:
        print("\n❌ El proceso no se completó correctamente")