#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para traducir columnas categóricas de un CSV usando diccionario de codificaciones
Traduce de números a texto usando el mapeo inverso del diccionario
"""

import pandas as pd
import json
import sys
import os
from pathlib import Path

def cargar_diccionario_codificaciones(ruta_diccionario):
    """
    Carga el diccionario de codificaciones desde un archivo JSON
    
    Args:
        ruta_diccionario (str): Ruta al archivo JSON con las codificaciones
    
    Returns:
        dict: Diccionario con las codificaciones
    """
    try:
        with open(ruta_diccionario, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_diccionario}")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo {ruta_diccionario} no es un JSON válido")
        return None

def crear_mapeo_inverso(mapeo_original):
    """
    Crea un mapeo inverso: de número a texto
    
    Args:
        mapeo_original (dict): Mapeo original de texto a número
    
    Returns:
        dict: Mapeo inverso de número a texto
    """
    return {v: k for k, v in mapeo_original.items()}

def traducir_csv(ruta_csv_entrada, ruta_diccionario, ruta_csv_salida=None):
    """
    Traduce las columnas especificadas de un CSV usando el diccionario de codificaciones
    Convierte números a texto usando el mapeo inverso
    
    Args:
        ruta_csv_entrada (str): Ruta del archivo CSV de entrada
        ruta_diccionario (str): Ruta del archivo JSON con las codificaciones
        ruta_csv_salida (str, optional): Ruta del archivo CSV de salida
    
    Returns:
        bool: True si la traducción fue exitosa, False en caso contrario
    """
    
    # Columnas a traducir
    columnas_a_traducir = [
        "Lugar de nacimiento (país)",
        "Nacionalidad",
        "Posición principal", 
        "Club actual",
        "Proveedor"
    ]
    
    # Cargar diccionario de codificaciones
    print("Cargando diccionario de codificaciones...")
    diccionario = cargar_diccionario_codificaciones(ruta_diccionario)
    if diccionario is None:
        return False
    
    # Leer CSV de entrada
    print(f"Leyendo archivo CSV: {ruta_csv_entrada}")
    try:
        df = pd.read_csv(ruta_csv_entrada, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_csv_entrada}")
        return False
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return False
    
    print(f"CSV cargado exitosamente. Shape: {df.shape}")
    print(f"Columnas encontradas: {list(df.columns)}")
    
    # Crear una copia del DataFrame para no modificar el original
    df_traducido = df.copy()
    
    # Traducir cada columna
    columnas_traducidas = []
    for columna in columnas_a_traducir:
        if columna in df_traducido.columns:
            if columna in diccionario:
                print(f"\nTraduciendo columna: {columna}")
                
                # Crear mapeo inverso (número -> texto)
                mapeo_original = diccionario[columna]["mapeo"]
                mapeo_inverso = crear_mapeo_inverso(mapeo_original)
                
                print(f"  - Mapeo disponible: {len(mapeo_inverso)} códigos")
                print(f"  - Rango de códigos: {min(mapeo_inverso.keys())} - {max(mapeo_inverso.keys())}")
                
                # Contar valores únicos antes de la traducción
                valores_unicos_antes = df_traducido[columna].nunique()
                valores_nulos_antes = df_traducido[columna].isnull().sum()
                
                # Mostrar algunos valores originales para debug
                valores_originales_muestra = df_traducido[columna].dropna().unique()[:10]
                print(f"  - Muestra de valores originales: {list(valores_originales_muestra)}")
                
                # Aplicar la traducción inversa (número -> texto)
                df_traducido[columna] = df_traducido[columna].map(mapeo_inverso)
                
                # Contar valores después de la traducción
                valores_nulos_despues = df_traducido[columna].isnull().sum()
                valores_no_mapeados = valores_nulos_despues - valores_nulos_antes
                
                print(f"  - Valores únicos originales: {valores_unicos_antes}")
                print(f"  - Valores nulos antes: {valores_nulos_antes}")
                print(f"  - Valores nulos después: {valores_nulos_despues}")
                print(f"  - Valores no mapeados: {valores_no_mapeados}")
                
                if valores_no_mapeados > 0:
                    # Mostrar valores que no se pudieron mapear
                    valores_originales_set = set(df[columna].dropna().astype(int).unique())
                    valores_en_diccionario_set = set(mapeo_inverso.keys())
                    valores_no_encontrados = valores_originales_set - valores_en_diccionario_set
                    
                    if valores_no_encontrados:
                        print(f"  - Códigos no encontrados en el diccionario: {sorted(list(valores_no_encontrados))}")
                
                # Mostrar algunos valores traducidos para verificación
                valores_traducidos_muestra = df_traducido[columna].dropna().unique()[:5]
                print(f"  - Muestra de valores traducidos: {list(valores_traducidos_muestra)}")
                
                columnas_traducidas.append(columna)
            else:
                print(f"Advertencia: La columna '{columna}' no se encontró en el diccionario de codificaciones")
        else:
            print(f"Advertencia: La columna '{columna}' no existe en el CSV")
    
    # Definir nombre del archivo de salida si no se especificó
    if ruta_csv_salida is None:
        ruta_entrada = Path(ruta_csv_entrada)
        nombre_base = ruta_entrada.stem
        extension = ruta_entrada.suffix
        ruta_csv_salida = ruta_entrada.parent / f"{nombre_base}_traducido{extension}"
    
    # Guardar CSV traducido
    print(f"\nGuardando archivo traducido: {ruta_csv_salida}")
    try:
        df_traducido.to_csv(ruta_csv_salida, index=False, encoding='utf-8')
        print("Archivo guardado exitosamente!")
        
        # Mostrar resumen
        print("\n=== RESUMEN DE LA TRADUCCIÓN ===")
        print(f"Archivo original: {ruta_csv_entrada}")
        print(f"Archivo traducido: {ruta_csv_salida}")
        print(f"Filas procesadas: {len(df_traducido)}")
        print(f"Columnas traducidas: {columnas_traducidas}")
        
        # Mostrar estadísticas de traducción por columna
        for columna in columnas_traducidas:
            valores_traducidos = df_traducido[columna].notna().sum()
            total_valores = len(df_traducido)
            porcentaje = (valores_traducidos / total_valores) * 100
            print(f"  - {columna}: {valores_traducidos}/{total_valores} valores traducidos ({porcentaje:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")
        return False

def main():
    """
    Función principal para ejecutar el script desde línea de comandos
    """
    if len(sys.argv) < 2:
        print("Uso: python traductor_csv.py <archivo_csv> [archivo_salida]")
        print("Ejemplo: python traductor_csv.py datos.csv datos_traducidos.csv")
        print("\nEste script traduce números a texto usando el diccionario de codificaciones.")
        sys.exit(1)
    
    ruta_csv_entrada = sys.argv[1]
    ruta_csv_salida = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Buscar el diccionario de codificaciones
    rutas_posibles_diccionario = [
        "diccionario_codificaciones.json",
        "web/data_preparation/diccionario_codificaciones.json",
        os.path.join(os.path.dirname(__file__), "diccionario_codificaciones.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "diccionario_codificaciones.json")
    ]
    
    ruta_diccionario = None
    for ruta in rutas_posibles_diccionario:
        if os.path.exists(ruta):
            ruta_diccionario = ruta
            break
    
    if ruta_diccionario is None:
        print("Error: No se encontró el archivo diccionario_codificaciones.json")
        print("Rutas buscadas:")
        for ruta in rutas_posibles_diccionario:
            print(f"  - {ruta}")
        sys.exit(1)
    
    print(f"Usando diccionario: {ruta_diccionario}")
    
    # Ejecutar traducción
    exito = traducir_csv(ruta_csv_entrada, ruta_diccionario, ruta_csv_salida)
    
    if exito:
        print("\n✅ Traducción completada exitosamente!")
    else:
        print("\n❌ Error durante la traducción")
        sys.exit(1)

if __name__ == "__main__":
    main() 