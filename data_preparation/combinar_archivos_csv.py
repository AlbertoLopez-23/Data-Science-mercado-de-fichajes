#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código auxiliar para combinar archivos CSV de una carpeta y calcular nuevas columnas
Creado para procesar datos de predicciones de valores de mercado
"""

import pandas as pd
import os
import glob
import numpy as np

def combinar_archivos_csv(carpeta_entrada, archivo_salida="datos_combinados.csv"):
    """
    Combina todos los archivos CSV de una carpeta en un solo archivo
    
    Parameters:
    -----------
    carpeta_entrada : str
        Ruta de la carpeta que contiene los archivos CSV
    archivo_salida : str
        Nombre del archivo de salida (default: "datos_combinados.csv")
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con todos los datos combinados y las nuevas columnas
    """
    
    # Verificar que la carpeta existe
    if not os.path.exists(carpeta_entrada):
        raise FileNotFoundError(f"La carpeta '{carpeta_entrada}' no existe")
    
    # Buscar todos los archivos CSV en la carpeta
    patron_csv = os.path.join(carpeta_entrada, "*.csv")
    archivos_csv = glob.glob(patron_csv)
    
    if not archivos_csv:
        raise FileNotFoundError(f"No se encontraron archivos CSV en la carpeta '{carpeta_entrada}'")
    
    print(f"Encontrados {len(archivos_csv)} archivos CSV")
    
    # Lista para almacenar los DataFrames
    dataframes = []
    
    # Leer cada archivo CSV
    for archivo in archivos_csv:
        print(f"Procesando: {os.path.basename(archivo)}")
        try:
            df = pd.read_csv(archivo)
            
            # Verificar que tiene las columnas necesarias
            if 'Valor de mercado actual (numérico)' in df.columns and 'Valor_Predicho' in df.columns:
                dataframes.append(df)
                print(f"  - Agregado: {len(df)} filas")
            else:
                print(f"  - OMITIDO: No tiene las columnas requeridas")
                
        except Exception as e:
            print(f"  - ERROR al leer {archivo}: {e}")
    
    if not dataframes:
        raise ValueError("No se pudieron leer archivos válidos")
    
    # Combinar todos los DataFrames
    print("\nCombinando datos...")
    df_combinado = pd.concat(dataframes, ignore_index=True)
    
    # Eliminar duplicados si los hay (basado en todas las columnas)
    filas_antes = len(df_combinado)
    df_combinado = df_combinado.drop_duplicates()
    filas_despues = len(df_combinado)
    
    if filas_antes != filas_despues:
        print(f"Se eliminaron {filas_antes - filas_despues} filas duplicadas")
    
    # Calcular nuevas columnas
    print("Calculando nuevas columnas...")
    df_combinado = calcular_nuevas_columnas(df_combinado)
    
    # Guardar archivo combinado
    df_combinado.to_csv(archivo_salida, index=False)
    print(f"\nArchivo guardado: {archivo_salida}")
    print(f"Total de filas: {len(df_combinado)}")
    print(f"Total de columnas: {len(df_combinado.columns)}")
    
    return df_combinado

def calcular_nuevas_columnas(df):
    """
    Calcula las nuevas columnas: diferencia y diferencia relativa (porcentual)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con las columnas 'Valor de mercado actual (numérico)' y 'Valor_Predicho'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las nuevas columnas agregadas
    """
    
    # Verificar que las columnas existen
    if 'Valor de mercado actual (numérico)' not in df.columns:
        raise ValueError("No se encontró la columna 'Valor de mercado actual (numérico)'")
    if 'Valor_Predicho' not in df.columns:
        raise ValueError("No se encontró la columna 'Valor_Predicho'")
    
    # Convertir a numérico si no lo están
    df['Valor de mercado actual (numérico)'] = pd.to_numeric(df['Valor de mercado actual (numérico)'], errors='coerce')
    df['Valor_Predicho'] = pd.to_numeric(df['Valor_Predicho'], errors='coerce')
    
    # Calcular diferencia: Valor real - Valor predicho
    df['Diferencia_Valor'] = df['Valor de mercado actual (numérico)'] - df['Valor_Predicho']
    
    # Calcular diferencia relativa (porcentual) manteniendo el signo
    # Positivo = predicción subestimó el valor real (valor real > predicción)
    # Negativo = predicción sobreestimó el valor real (valor real < predicción)
    df['Diferencia_Relativa_Pct'] = ((df['Valor de mercado actual (numérico)'] - df['Valor_Predicho']) / 
                                     df['Valor de mercado actual (numérico)']) * 100
    
    # Manejar casos donde el valor real es 0 (evitar división por cero)
    mask_cero = df['Valor de mercado actual (numérico)'] == 0
    if mask_cero.any():
        print(f"Advertencia: {mask_cero.sum()} filas tienen valor real = 0, se asigna NaN a la diferencia relativa")
        df.loc[mask_cero, 'Diferencia_Relativa_Pct'] = np.nan
    
    # Calcular estadísticas de la diferencia
    print("\nEstadísticas de la diferencia (Valor real - Valor predicho):")
    print(f"Media: {df['Diferencia_Valor'].mean():.2f}")
    print(f"Mediana: {df['Diferencia_Valor'].median():.2f}")
    print(f"Desviación estándar: {df['Diferencia_Valor'].std():.2f}")
    print(f"Mínimo: {df['Diferencia_Valor'].min():.2f}")
    print(f"Máximo: {df['Diferencia_Valor'].max():.2f}")
    
    # Estadísticas de la diferencia relativa
    print(f"\nEstadísticas de la diferencia relativa (%):")
    print(f"Media: {df['Diferencia_Relativa_Pct'].mean():.2f}%")
    print(f"Mediana: {df['Diferencia_Relativa_Pct'].median():.2f}%")
    print(f"Desviación estándar: {df['Diferencia_Relativa_Pct'].std():.2f}%")
    print(f"Mínimo: {df['Diferencia_Relativa_Pct'].min():.2f}%")
    print(f"Máximo: {df['Diferencia_Relativa_Pct'].max():.2f}%")
    
    # Análisis de dirección de los errores
    sobreestimaciones = (df['Diferencia_Relativa_Pct'] < 0).sum()
    subestimaciones = (df['Diferencia_Relativa_Pct'] > 0).sum()
    exactos = (df['Diferencia_Relativa_Pct'] == 0).sum()
    
    print(f"\nAnálisis de predicciones:")
    print(f"Sobreestimaciones (predicción > real): {sobreestimaciones} ({sobreestimaciones/len(df)*100:.1f}%)")
    print(f"Subestimaciones (predicción < real): {subestimaciones} ({subestimaciones/len(df)*100:.1f}%)")
    print(f"Predicciones exactas: {exactos} ({exactos/len(df)*100:.1f}%)")
    
    # Rangos de error relativo
    print(f"\nDistribución de errores relativos:")
    print(f"Error < 5%: {(np.abs(df['Diferencia_Relativa_Pct']) < 5).sum()} casos")
    print(f"Error 5-10%: {((np.abs(df['Diferencia_Relativa_Pct']) >= 5) & (np.abs(df['Diferencia_Relativa_Pct']) < 10)).sum()} casos")
    print(f"Error 10-25%: {((np.abs(df['Diferencia_Relativa_Pct']) >= 10) & (np.abs(df['Diferencia_Relativa_Pct']) < 25)).sum()} casos")
    print(f"Error >= 25%: {(np.abs(df['Diferencia_Relativa_Pct']) >= 25).sum()} casos")
    
    return df

def deshacer_standard_scaler(df, columnas_escaladas, archivo_original=None, media_y_escala=None):
    """
    Deshace la transformación StandardScaler en las columnas especificadas
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con las columnas escaladas que se quieren desnormalizar
    columnas_escaladas : list
        Lista de nombres de columnas que fueron escaladas con StandardScaler
    archivo_original : str, optional
        Ruta al archivo original (antes de la normalización) para calcular parámetros
    media_y_escala : dict, optional
        Diccionario con los parámetros {columna: {'mean': valor, 'scale': valor}}
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las columnas desnormalizadas
    """
    
    df_desnormalizado = df.copy()
    
    # Verificar que las columnas existen
    columnas_disponibles = [col for col in columnas_escaladas if col in df.columns]
    columnas_faltantes = [col for col in columnas_escaladas if col not in df.columns]
    
    if columnas_faltantes:
        print(f"⚠️  Las siguientes columnas no se encontraron: {columnas_faltantes}")
    
    if not columnas_disponibles:
        print("❌ No se encontraron columnas válidas para desnormalizar")
        return df_desnormalizado
    
    print(f"🔄 Deshaciendo StandardScaler en {len(columnas_disponibles)} columnas...")
    
    # Si se proporcionan los parámetros directamente
    if media_y_escala:
        for columna in columnas_disponibles:
            if columna in media_y_escala:
                mean_val = media_y_escala[columna]['mean']
                scale_val = media_y_escala[columna]['scale']
                
                # Fórmula inversa del StandardScaler: valor_original = (valor_escalado * scale) + mean
                df_desnormalizado[columna] = (df[columna] * scale_val) + mean_val
                print(f"  ✓ {columna}: mean={mean_val:.4f}, scale={scale_val:.4f}")
    
    # Si se proporciona el archivo original, calcular parámetros
    elif archivo_original and os.path.exists(archivo_original):
        print(f"📂 Cargando archivo original: {archivo_original}")
        df_original = pd.read_csv(archivo_original)
        
        for columna in columnas_disponibles:
            if columna in df_original.columns:
                # Calcular media y desviación estándar del archivo original
                datos_originales = df_original[columna].replace([np.inf, -np.inf], np.nan)
                mean_val = datos_originales.mean()
                std_val = datos_originales.std()
                
                # Desnormalizar: valor_original = (valor_escalado * std) + mean
                df_desnormalizado[columna] = (df[columna] * std_val) + mean_val
                print(f"  ✓ {columna}: mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                print(f"  ⚠️  {columna} no encontrada en archivo original")
    
    else:
        print("❌ Se necesita proporcionar 'archivo_original' o 'media_y_escala'")
        print("   Para usar 'media_y_escala', proporciona un diccionario como:")
        print("   {'columna': {'mean': valor_medio, 'scale': valor_escala}}")
        return df_desnormalizado
    
    print("✅ Desnormalización completada")
    return df_desnormalizado

def desnormalizar_archivos_csv(archivo_csv, archivo_original=None, archivo_salida=None):
    """
    Función principal para desnormalizar archivos CSV
    
    Parameters:
    -----------
    archivo_csv : str
        Ruta del archivo CSV con datos normalizados
    archivo_original : str, optional
        Ruta del archivo original (antes de normalización)
    archivo_salida : str, optional
        Nombre del archivo de salida (default: agregar '_desnormalizado' al nombre)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las columnas desnormalizadas
    """
    
    # Columnas que fueron escaladas con StandardScaler según el usuario
    columnas_escaladas = [
        'comprado_por', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 
        'gk_reflexes', 'overallrating', 'potential', 'jumping', 'understat_goals', 
        'strength', 'vision', 'headingaccuracy', 'ballcontrol', 'acceleration', 
        'interceptions', 'understat_matches', 'reactions', 'longpassing', 'dribbling', 
        'understat_assists', 'standingtackle', 'understat_minutes', 'penalties'
    ]
    
    print("🚀 INICIANDO DESNORMALIZACIÓN DE STANDARDSCALER")
    print("=" * 60)
    
    # Verificar que el archivo existe
    if not os.path.exists(archivo_csv):
        raise FileNotFoundError(f"El archivo '{archivo_csv}' no existe")
    
    # Cargar el archivo CSV
    print(f"📂 Cargando archivo: {archivo_csv}")
    df = pd.read_csv(archivo_csv)
    print(f"   - Filas: {len(df)}")
    print(f"   - Columnas: {len(df.columns)}")
    
    # Mostrar columnas escaladas disponibles
    columnas_disponibles = [col for col in columnas_escaladas if col in df.columns]
    print(f"\n📋 Columnas escaladas encontradas: {len(columnas_disponibles)}")
    for col in columnas_disponibles:
        print(f"   ✓ {col}")
    
    columnas_faltantes = [col for col in columnas_escaladas if col not in df.columns]
    if columnas_faltantes:
        print(f"\n⚠️  Columnas escaladas no encontradas: {len(columnas_faltantes)}")
        for col in columnas_faltantes:
            print(f"   ✗ {col}")
    
    # Desnormalizar
    df_desnormalizado = deshacer_standard_scaler(
        df, 
        columnas_escaladas, 
        archivo_original=archivo_original
    )
    
    # Generar nombre de archivo de salida si no se proporciona
    if archivo_salida is None:
        base_name = os.path.splitext(os.path.basename(archivo_csv))[0]
        archivo_salida = f"{base_name}_desnormalizado.csv"
    
    # Guardar archivo desnormalizado
    df_desnormalizado.to_csv(archivo_salida, index=False)
    
    print(f"\n✅ PROCESO COMPLETADO")
    print(f"📁 Archivo guardado: {archivo_salida}")
    print(f"📊 Filas procesadas: {len(df_desnormalizado)}")
    print(f"🔧 Columnas desnormalizadas: {len(columnas_disponibles)}")
    
    return df_desnormalizado

def main():
    """
    Función principal - ejemplo de uso
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Combinar archivos CSV, calcular diferencias y desnormalizar StandardScaler')
    
    # Crear subcomandos
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponibles')
    
    # Subcomando para combinar archivos
    parser_combinar = subparsers.add_parser('combinar', help='Combinar archivos CSV de una carpeta')
    parser_combinar.add_argument('carpeta', help='Carpeta que contiene los archivos CSV')
    parser_combinar.add_argument('-o', '--output', default='datos_combinados.csv', help='Archivo de salida')
    
    # Subcomando para desnormalizar
    parser_deshacer = subparsers.add_parser('desnormalizar', help='Deshacer StandardScaler en archivo CSV')
    parser_deshacer.add_argument('archivo', help='Archivo CSV con datos normalizados')
    parser_deshacer.add_argument('-orig', '--original', help='Archivo original (antes de normalización)')
    parser_deshacer.add_argument('-o', '--output', help='Archivo de salida')
    
    args = parser.parse_args()
    
    try:
        if args.comando == 'combinar':
            df_resultado = combinar_archivos_csv(args.carpeta, args.output)
            
            # Mostrar información adicional
            print("\nPrimeras 5 filas del resultado:")
            print(df_resultado[['Nombre completo', 'Valor de mercado actual (numérico)', 
                              'Valor_Predicho', 'Diferencia_Valor', 'Diferencia_Relativa_Pct']].head())
            
            print(f"\nColumnas en el archivo final:")
            for i, col in enumerate(df_resultado.columns, 1):
                print(f"{i:2d}. {col}")
        
        elif args.comando == 'desnormalizar':
            df_resultado = desnormalizar_archivos_csv(
                args.archivo, 
                archivo_original=args.original,
                archivo_salida=args.output
            )
            
            # Mostrar estadísticas básicas de las columnas desnormalizadas
            columnas_escaladas = [
                'comprado_por', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 
                'gk_reflexes', 'overallrating', 'potential', 'jumping', 'understat_goals', 
                'strength', 'vision', 'headingaccuracy', 'ballcontrol', 'acceleration', 
                'interceptions', 'understat_matches', 'reactions', 'longpassing', 'dribbling', 
                'understat_assists', 'standingtackle', 'understat_minutes', 'penalties'
            ]
            
            columnas_disponibles = [col for col in columnas_escaladas if col in df_resultado.columns]
            if columnas_disponibles:
                print(f"\n📊 ESTADÍSTICAS DE COLUMNAS DESNORMALIZADAS:")
                print("=" * 50)
                for col in columnas_disponibles[:5]:  # Mostrar solo las primeras 5
                    stats = df_resultado[col].describe()
                    print(f"{col}:")
                    print(f"  Media: {stats['mean']:.2f}")
                    print(f"  Min: {stats['min']:.2f}")
                    print(f"  Max: {stats['max']:.2f}")
                    print()
        
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Ejemplo de uso directo:
# df_combinado = combinar_archivos_csv("XGBoost_09_db_porteros_filtered_top40pct") 