#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para deshacer la normalización StandardScaler usando el archivo original como referencia
"""

import pandas as pd
import numpy as np
import os
import sys

def desnormalizar_con_archivo_original(archivo_normalizado, archivo_original, archivo_salida=None):
    """
    Deshace la normalización StandardScaler usando el archivo original como referencia
    
    Parameters:
    -----------
    archivo_normalizado : str
        Ruta del archivo CSV con datos normalizados
    archivo_original : str
        Ruta del archivo original (antes de normalización)
    archivo_salida : str, optional
        Nombre del archivo de salida
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las columnas desnormalizadas
    """
    
    # Columnas que fueron escaladas con StandardScaler
    columnas_escaladas = [
        'comprado_por', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 
        'gk_reflexes', 'overallrating', 'potential', 'jumping', 'understat_goals', 
        'strength', 'vision', 'headingaccuracy', 'ballcontrol', 'acceleration', 
        'interceptions', 'understat_matches', 'reactions', 'longpassing', 'dribbling', 
        'understat_assists', 'standingtackle', 'understat_minutes', 'penalties'
    ]
    
    print("🚀 INICIANDO DESNORMALIZACIÓN DE STANDARDSCALER")
    print("=" * 60)
    
    # Verificar que los archivos existen
    if not os.path.exists(archivo_normalizado):
        raise FileNotFoundError(f"El archivo normalizado '{archivo_normalizado}' no existe")
    
    if not os.path.exists(archivo_original):
        raise FileNotFoundError(f"El archivo original '{archivo_original}' no existe")
    
    # Cargar archivos
    print(f"📂 Cargando archivo normalizado: {archivo_normalizado}")
    df_normalizado = pd.read_csv(archivo_normalizado)
    print(f"   - Filas: {len(df_normalizado)}")
    print(f"   - Columnas: {len(df_normalizado.columns)}")
    
    print(f"📂 Cargando archivo original: {archivo_original}")
    df_original = pd.read_csv(archivo_original)
    print(f"   - Filas: {len(df_original)}")
    print(f"   - Columnas: {len(df_original.columns)}")
    
    # Crear copia para desnormalizar
    df_desnormalizado = df_normalizado.copy()
    
    # Verificar columnas disponibles
    columnas_disponibles = []
    columnas_faltantes_norm = []
    columnas_faltantes_orig = []
    
    for col in columnas_escaladas:
        if col in df_normalizado.columns and col in df_original.columns:
            columnas_disponibles.append(col)
        elif col not in df_normalizado.columns:
            columnas_faltantes_norm.append(col)
        elif col not in df_original.columns:
            columnas_faltantes_orig.append(col)
    
    print(f"\n📋 ANÁLISIS DE COLUMNAS:")
    print(f"✅ Columnas disponibles para desnormalizar: {len(columnas_disponibles)}")
    for col in columnas_disponibles:
        print(f"   ✓ {col}")
    
    if columnas_faltantes_norm:
        print(f"\n⚠️  Columnas no encontradas en archivo normalizado: {len(columnas_faltantes_norm)}")
        for col in columnas_faltantes_norm:
            print(f"   ✗ {col}")
    
    if columnas_faltantes_orig:
        print(f"\n⚠️  Columnas no encontradas en archivo original: {len(columnas_faltantes_orig)}")
        for col in columnas_faltantes_orig:
            print(f"   ✗ {col}")
    
    if not columnas_disponibles:
        print("❌ No se encontraron columnas válidas para desnormalizar")
        return df_desnormalizado
    
    # Desnormalizar cada columna
    print(f"\n🔄 DESNORMALIZANDO {len(columnas_disponibles)} COLUMNAS:")
    print("-" * 50)
    
    parametros_usados = {}
    
    for columna in columnas_disponibles:
        # Calcular parámetros del archivo original
        datos_originales = df_original[columna].replace([np.inf, -np.inf], np.nan)
        
        # Calcular media y desviación estándar (parámetros del StandardScaler)
        mean_original = datos_originales.mean()
        std_original = datos_originales.std()
        
        # Verificar que no hay problemas con los parámetros
        if pd.isna(mean_original) or pd.isna(std_original) or std_original == 0:
            print(f"  ⚠️  {columna}: Parámetros inválidos (mean={mean_original}, std={std_original})")
            continue
        
        # Aplicar la fórmula inversa del StandardScaler
        # StandardScaler: (x - mean) / std
        # Inversa: x_original = (x_scaled * std) + mean
        df_desnormalizado[columna] = (df_normalizado[columna] * std_original) + mean_original
        
        # Guardar parámetros para referencia
        parametros_usados[columna] = {
            'mean': mean_original,
            'std': std_original
        }
        
        # Mostrar estadísticas
        valores_norm = df_normalizado[columna]
        valores_desnorm = df_desnormalizado[columna]
        
        print(f"  ✓ {columna}:")
        print(f"     Original -> Media: {mean_original:.4f}, Std: {std_original:.4f}")
        print(f"     Normalizado -> Media: {valores_norm.mean():.4f}, Std: {valores_norm.std():.4f}")
        print(f"     Desnormalizado -> Media: {valores_desnorm.mean():.4f}, Std: {valores_desnorm.std():.4f}")
        print()
    
    # Generar nombre de archivo de salida
    if archivo_salida is None:
        base_name = os.path.splitext(os.path.basename(archivo_normalizado))[0]
        archivo_salida = f"{base_name}_desnormalizado.csv"
    
    # Guardar archivo desnormalizado
    df_desnormalizado.to_csv(archivo_salida, index=False)
    
    print("✅ PROCESO COMPLETADO")
    print("=" * 60)
    print(f"📁 Archivo guardado: {archivo_salida}")
    print(f"📊 Filas procesadas: {len(df_desnormalizado)}")
    print(f"🔧 Columnas desnormalizadas: {len(parametros_usados)}")
    
    # Mostrar resumen de parámetros
    print(f"\n📋 RESUMEN DE PARÁMETROS UTILIZADOS:")
    print("-" * 50)
    for col, params in parametros_usados.items():
        print(f"{col}: mean={params['mean']:.4f}, std={params['std']:.4f}")
    
    return df_desnormalizado, parametros_usados

def main():
    """
    Función principal
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Deshacer StandardScaler usando archivo original como referencia')
    parser.add_argument('archivo_normalizado', help='Archivo CSV con datos normalizados')
    parser.add_argument('-orig', '--original', 
                       default='Datos/DB_unidas/03_db_columnas_ordenadas.csv',
                       help='Archivo original (default: DB_viejas/03_db_columnas_ordenadas.csv)')
    parser.add_argument('-o', '--output', help='Archivo de salida')
    
    args = parser.parse_args()
    
    try:
        df_resultado, parametros = desnormalizar_con_archivo_original(
            args.archivo_normalizado,
            args.original,
            args.output
        )
        
        print(f"\n🎉 ¡Desnormalización completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Ejemplo de uso directo:
# python desnormalizar_datos.py archivo_normalizado.csv
# python desnormalizar_datos.py archivo_normalizado.csv -orig DB_viejas/03_db_columnas_ordenadas.csv -o resultado_desnormalizado.csv 