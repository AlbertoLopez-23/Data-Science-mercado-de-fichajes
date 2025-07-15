import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuración para mejor visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Crear carpeta de resultados finales
def create_results_folder():
    """Crear carpeta para almacenar todos los resultados"""
    folder_name = "resultados_finales"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ Carpeta '{folder_name}' creada exitosamente")
    else:
        print(f"📁 Carpeta '{folder_name}' ya existe")
    return folder_name

# Configurar logging
def setup_logging(results_folder):
    """Configurar logging para guardar toda la salida"""
    import sys
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_folder, f"log_analisis_{timestamp}.txt")
    
    class Logger:
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log = open(log_file, "w", encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    print(f"📝 Log guardándose en: {log_file}")
    return log_file

def load_and_clean_data(file_path):
    """Cargar y limpiar los datos"""
    print("🚀 INICIANDO ANÁLISIS EXPLORATORIO DE DATOS - FÚTBOL")
    print("="*80)
    print("📊 Cargando datos...")
    
    df = pd.read_csv(file_path)
    
    print(f"📈 Dimensiones del dataset: {df.shape}")
    print(f"📋 Número de jugadores: {df.shape[0]:,}")
    print(f"📋 Número de variables: {df.shape[1]}")
    print(f"📋 Columnas disponibles: {df.columns.tolist()}")
    
    # Información básica sobre valores nulos
    print(f"\n🔍 Valores nulos por columna (primeras 10):")
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    print(missing_summary.head(10))
    
    return df

def basic_eda(df, results_folder):
    """Análisis exploratorio básico"""
    print("\n" + "="*80)
    print("📊 ANÁLISIS EXPLORATORIO GENERAL")
    print("="*80)
    
    # Información básica del dataset
    print("\n--- 📋 Información general del dataset ---")
    print(df.info())
    
    # Estadísticas descriptivas de variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n--- 📊 Variables numéricas ({len(numeric_cols)} variables) ---")
    print(df[numeric_cols].describe())
    
    # Variables categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\n--- 📝 Variables categóricas ({len(categorical_cols)} variables) ---")
    for col in categorical_cols[:5]:
        print(f"   {col}: {df[col].nunique()} valores únicos")
    
    # Valores faltantes detallados
    print("\n--- ❌ Valores faltantes ---")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Variable': missing_data.index,
        'Valores_Faltantes': missing_data.values,
        'Porcentaje': missing_percent.values
    }).sort_values('Porcentaje', ascending=False)
    
    print(missing_df[missing_df['Porcentaje'] > 0].head(10))
    
    # Guardar información básica
    basic_info_file = os.path.join(results_folder, "informacion_basica.csv")
    missing_df.to_csv(basic_info_file, index=False)
    print(f"💾 Información básica guardada en: {basic_info_file}")
    
    # Información sobre valores de mercado
    if 'Valor de mercado actual (numérico)' in df.columns and 'Valor_Predicho' in df.columns:
        print(f"\n💰 VALORES DE MERCADO:")
        print(f"   💰 Valor real - Media: €{df['Valor de mercado actual (numérico)'].mean():,.0f}")
        print(f"   💰 Valor real - Mediana: €{df['Valor de mercado actual (numérico)'].median():,.0f}")
        print(f"   🎯 Valor predicho - Media: €{df['Valor_Predicho'].mean():,.0f}")
        print(f"   🎯 Valor predicho - Mediana: €{df['Valor_Predicho'].median():,.0f}")
    
    # Top equipos por número de jugadores
    if 'Club actual' in df.columns:
        print(f"\n🏆 TOP 10 EQUIPOS POR NÚMERO DE JUGADORES:")
        top_teams = df['Club actual'].value_counts().head(10)
        print(top_teams)
    
    # Top nacionalidades
    if 'Lugar de nacimiento (país)' in df.columns:
        print(f"\n🌍 TOP 10 NACIONALIDADES:")
        top_nationalities = df['Lugar de nacimiento (país)'].value_counts().head(10)
        print(top_nationalities)
    
    return numeric_cols, categorical_cols

def calculate_differences(df):
    """Calcular diferencias entre valor predicho y valor real"""
    print("\n" + "="*80)
    print("📊 CÁLCULO DE DIFERENCIAS VALOR PREDICHO VS VALOR REAL")
    print("="*80)
    
    if 'Valor de mercado actual (numérico)' in df.columns and 'Valor_Predicho' in df.columns:
        # Calcular diferencias: VALOR PREDICHO - VALOR REAL
        df['Diferencia_Absoluta'] = df['Valor_Predicho'] - df['Valor de mercado actual (numérico)']
        df['Diferencia_Relativa'] = (df['Diferencia_Absoluta'] / df['Valor de mercado actual (numérico)']) * 100
        
        # Reemplazar infinitos y NaN
        df['Diferencia_Relativa'] = df['Diferencia_Relativa'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['Diferencia_Relativa'])
        
        # Mostrar estadísticas antes del filtrado
        print(f"\n--- 📊 Estadísticas ANTES del filtrado ---")
        print(f"   📊 Total de instancias: {len(df):,}")
        print(f"   📊 Diferencia absoluta promedio: €{df['Diferencia_Absoluta'].mean():,.2f}")
        print(f"   📊 Diferencia relativa promedio: {df['Diferencia_Relativa'].mean():.2f}%")
        print(f"   📊 Error absoluto medio: €{abs(df['Diferencia_Absoluta']).mean():,.0f}")
        print(f"   📊 Error relativo medio: {abs(df['Diferencia_Relativa']).mean():.2f}%")
        print(f"   🟢 Predicciones superiores al valor real: {(df['Diferencia_Absoluta'] > 0).sum():,} ({(df['Diferencia_Absoluta'] > 0).mean()*100:.1f}%)")
        print(f"   🔴 Predicciones inferiores al valor real: {(df['Diferencia_Absoluta'] < 0).sum():,} ({(df['Diferencia_Absoluta'] < 0).mean()*100:.1f}%)")
        
        # FILTRAR instancias con diferencia relativa superior al 25%
        print(f"\n--- 🔍 APLICANDO FILTRO: Diferencia relativa ≤ 25% ---")
        instancias_originales = len(df)
        
        # Filtrar por diferencia relativa absoluta <= 25%
        df_filtered = df[abs(df['Diferencia_Relativa']) <= 25].copy()
        
        instancias_filtradas = len(df_filtered)
        instancias_eliminadas = instancias_originales - instancias_filtradas
        porcentaje_eliminado = (instancias_eliminadas / instancias_originales) * 100
        
        print(f"   ✅ Instancias originales: {instancias_originales:,}")
        print(f"   ❌ Instancias eliminadas: {instancias_eliminadas:,} ({porcentaje_eliminado:.1f}%)")
        print(f"   ✅ Instancias restantes: {instancias_filtradas:,} ({100-porcentaje_eliminado:.1f}%)")
        
        # Mostrar estadísticas de las instancias eliminadas
        if instancias_eliminadas > 0:
            df_eliminadas = df[abs(df['Diferencia_Relativa']) > 25]
            print(f"\n--- 🗑️ Estadísticas de instancias ELIMINADAS ---")
            print(f"   📊 Diferencia relativa media: {df_eliminadas['Diferencia_Relativa'].mean():.2f}%")
            print(f"   📊 Diferencia relativa mínima: {df_eliminadas['Diferencia_Relativa'].min():.2f}%")
            print(f"   📊 Diferencia relativa máxima: {df_eliminadas['Diferencia_Relativa'].max():.2f}%")
        
        # Estadísticas después del filtrado
        print(f"\n--- 📊 Estadísticas DESPUÉS del filtrado ---")
        print(f"   📊 Diferencia absoluta promedio: €{df_filtered['Diferencia_Absoluta'].mean():,.2f}")
        print(f"   📊 Diferencia absoluta mediana: €{df_filtered['Diferencia_Absoluta'].median():,.2f}")
        print(f"   📊 Desviación estándar: €{df_filtered['Diferencia_Absoluta'].std():,.2f}")
        print(f"   📊 Diferencia relativa promedio: {df_filtered['Diferencia_Relativa'].mean():.2f}%")
        print(f"   📊 Diferencia relativa mediana: {df_filtered['Diferencia_Relativa'].median():.2f}%")
        print(f"   📊 Error absoluto medio: €{abs(df_filtered['Diferencia_Absoluta']).mean():,.0f}")
        print(f"   📊 Error relativo medio: {abs(df_filtered['Diferencia_Relativa']).mean():.2f}%")
        print(f"   🟢 Predicciones superiores al valor real (filtrado): {(df_filtered['Diferencia_Absoluta'] > 0).sum():,} ({(df_filtered['Diferencia_Absoluta'] > 0).mean()*100:.1f}%)")
        print(f"   🔴 Predicciones inferiores al valor real (filtrado): {(df_filtered['Diferencia_Absoluta'] < 0).sum():,} ({(df_filtered['Diferencia_Absoluta'] < 0).mean()*100:.1f}%)")
        
        # Agregar columna indicadora del filtrado al dataset original
        df['Filtrado_25_pct'] = abs(df['Diferencia_Relativa']) <= 25
        
        # Retornar el dataset filtrado para análisis posteriores
        return df_filtered
        
    return df

def correlation_analysis(df, results_folder):
    """Análisis de correlaciones completo"""
    print("\n" + "="*80)
    print("🔗 ANÁLISIS DE CORRELACIONES")
    print("="*80)
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calcular correlaciones con valor predicho
    correlations = []
    for col in numeric_cols:
        if col != 'Valor_Predicho' and col != 'Valor de mercado actual (numérico)' and not df[col].isna().all():
            try:
                corr, p_value = pearsonr(df[col].fillna(df[col].median()), 
                                       df['Valor_Predicho'].fillna(df['Valor_Predicho'].median()))
                correlations.append({
                    'Variable': col,
                    'Correlacion': corr,
                    'P_value': p_value,
                    'Correlacion_Abs': abs(corr)
                })
            except:
                continue
    
    correlations_df = pd.DataFrame(correlations).sort_values('Correlacion_Abs', ascending=False)
    
    print("\n--- 🔗 Top 20 variables más correlacionadas con Valor Predicho ---")
    print(correlations_df.head(20)[['Variable', 'Correlacion']].round(4))
    
    # Guardar correlaciones
    correlations_file = os.path.join(results_folder, "correlaciones_variables.csv")
    correlations_df.to_csv(correlations_file, index=False)
    print(f"💾 Correlaciones guardadas en: {correlations_file}")
    
    return correlations_df

def create_visualizations(df, correlations_df, results_folder):
    """Crear todas las visualizaciones"""
    print("\n" + "="*80)
    print("📊 GENERANDO VISUALIZACIONES...")
    print("="*80)
    print("⚠️  NOTA: Las visualizaciones se basan en datos filtrados (diferencia relativa ≤ 25%)")
    print("🟢 Verde: Predicción > Valor Real | 🔴 Rojo: Predicción < Valor Real")
    
    # Variables más correlacionadas para visualizaciones
    top_vars = correlations_df.head(15)['Variable'].tolist()
    top_vars.append('Valor_Predicho')
    
    # Crear figura principal con 4 subplots (2x2)
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Mapa de calor de correlaciones
    plt.subplot(2, 2, 1)
    correlation_matrix = df[top_vars].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    plt.title('Mapa de Calor: Variables más Correlacionadas con Valor Predicho', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 2. Distribución de diferencias relativas con colores
    plt.subplot(2, 2, 2)
    positive_rel_diffs = df[df['Diferencia_Relativa'] >= 0]['Diferencia_Relativa']
    negative_rel_diffs = df[df['Diferencia_Relativa'] < 0]['Diferencia_Relativa']
    
    plt.hist(positive_rel_diffs, bins=25, alpha=0.7, color='green', edgecolor='black', label=f'Pred > Real ({len(positive_rel_diffs)})')
    plt.hist(negative_rel_diffs, bins=25, alpha=0.7, color='red', edgecolor='black', label=f'Pred < Real ({len(negative_rel_diffs)})')
    plt.axvline(df['Diferencia_Relativa'].mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Media: {df["Diferencia_Relativa"].mean():.1f}%')
    plt.axvline(df['Diferencia_Relativa'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Mediana: {df["Diferencia_Relativa"].median():.1f}%')
    plt.xlabel('Diferencia Relativa (%) [Predicho - Real]')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Diferencias Relativas', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Gráfico de barras con variables más correlacionadas (verde para positivo, rojo para negativo)
    plt.subplot(2, 2, 3)
    top_corr = correlations_df.head(10)
    colors = ['green' if x >= 0 else 'red' for x in top_corr['Correlacion']]
    bars = plt.barh(range(len(top_corr)), top_corr['Correlacion'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_corr)), top_corr['Variable'])
    plt.xlabel('Correlación con Valor Predicho')
    plt.title('Top 10 Variables más Correlacionadas', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores en las barras
    for i, (bar, value) in enumerate(zip(bars, top_corr['Correlacion'])):
        plt.text(value + 0.01 if value > 0 else value - 0.01, i, f'{value:.3f}', 
                 va='center', ha='left' if value > 0 else 'right', fontsize=9)
    
    # 4. Scatter plot: Valor Real vs Valor Predicho con colores
    plt.subplot(2, 2, 4)
    # Colorear puntos según si la predicción es mayor o menor que el valor real
    colors_scatter = ['green' if pred > real else 'red' 
                     for pred, real in zip(df['Valor_Predicho'], df['Valor de mercado actual (numérico)'])]
    plt.scatter(df['Valor_Predicho'], df['Valor de mercado actual (numérico)'], 
               c=colors_scatter, alpha=0.6, s=30)
    min_val = min(df['Valor_Predicho'].min(), df['Valor de mercado actual (numérico)'].min())
    max_val = max(df['Valor_Predicho'].max(), df['Valor de mercado actual (numérico)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Predicción Perfecta')
    plt.xlabel('Valor Predicho (€)')
    plt.ylabel('Valor Real (€)')
    plt.title('Valor Real vs Valor Predicho\n🟢 Pred > Real | 🔴 Pred < Real', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar visualización general
    general_viz_file = os.path.join(results_folder, "analisis_eda_general.png")
    plt.savefig(general_viz_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Visualización general guardada en: {general_viz_file}")

def team_analysis(df, results_folder):
    """Análisis completo por equipos"""
    print("\n" + "="*80)
    print("⚽ ANÁLISIS POR EQUIPOS DE FÚTBOL")
    print("="*80)
    print("⚠️  NOTA: Análisis basado en datos filtrados (diferencia relativa ≤ 25%)")
    
    if 'Club actual' not in df.columns:
        print("❌ No se encontró la columna 'Club actual'")
        return None
    
    # Estadísticas por equipo
    equipos_stats = df.groupby('Club actual').agg({
        'Diferencia_Relativa': ['mean', 'median', 'std', 'count'],
        'Diferencia_Absoluta': ['mean', 'median'],
        'Valor de mercado actual (numérico)': 'mean',
        'Valor_Predicho': 'mean'
    }).round(2)
    
    equipos_stats.columns = ['Dif_Rel_Media', 'Dif_Rel_Mediana', 'Dif_Rel_Std', 'Num_Jugadores',
                            'Dif_Abs_Media', 'Dif_Abs_Mediana', 'Valor_Real_Promedio', 'Valor_Pred_Promedio']
    
    # Filtrar equipos con al menos 3 jugadores
    equipos_stats = equipos_stats[equipos_stats['Num_Jugadores'] >= 3]
    
    # Ordenar por mayor diferencia relativa absoluta
    equipos_stats['Dif_Rel_Abs'] = abs(equipos_stats['Dif_Rel_Media'])
    equipos_stats = equipos_stats.sort_values('Dif_Rel_Abs', ascending=False)
    
    print("\n--- 🏆 Top 15 equipos con mayor diferencia relativa absoluta ---")
    print(equipos_stats.head(15)[['Dif_Rel_Media', 'Num_Jugadores']])
    
    # Crear visualización con solo un gráfico - Top 10 por diferencia relativa absoluta
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Top 10 equipos por diferencia relativa absoluta
    top_10 = equipos_stats.head(10)
    colors = ['green' if x >= 0 else 'red' for x in top_10['Dif_Rel_Media']]
    
    bars = ax.barh(range(len(top_10)), top_10['Dif_Rel_Media'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index, fontsize=11)
    ax.set_xlabel('Diferencia Relativa Media (%)', fontsize=12)
    ax.set_title('Top 10 Equipos por Mayor Diferencia Relativa\n🟢 Pred > Real | 🔴 Pred < Real', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Calcular límites del eje X para acomodar las etiquetas
    min_val = min(top_10['Dif_Rel_Media'])
    max_val = max(top_10['Dif_Rel_Media'])
    x_range = max_val - min_val
    ax.set_xlim(min_val - x_range * 0.3, max_val + x_range * 0.3)
    
    # Añadir valores en las barras con mejor espaciado
    for i, (bar, value) in enumerate(zip(bars, top_10['Dif_Rel_Media'])):
        ax.text(value + 1.0 if value > 0 else value - 1.0, i, f'{value:.1f}%', 
                va='center', ha='left' if value > 0 else 'right', fontsize=10, fontweight='bold')
    
    # Ajustar márgenes para evitar solapamiento
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    
    # Guardar análisis de equipos
    equipos_viz_file = os.path.join(results_folder, "analisis_equipos.png")
    plt.savefig(equipos_viz_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Análisis de equipos guardado en: {equipos_viz_file}")
    
    # Guardar estadísticas de equipos
    equipos_stats_file = os.path.join(results_folder, "resultados_equipos.csv")
    equipos_stats.to_csv(equipos_stats_file)
    print(f"💾 Estadísticas de equipos guardadas en: {equipos_stats_file}")
    
    return equipos_stats

def nationality_analysis(df, results_folder):
    """Análisis completo por nacionalidades"""
    print("\n" + "="*80)
    print("🌍 ANÁLISIS POR NACIONALIDADES")
    print("="*80)
    print("⚠️  NOTA: Análisis basado en datos filtrados (diferencia relativa ≤ 25%)")
    
    if 'Lugar de nacimiento (país)' not in df.columns:
        print("❌ No se encontró la columna 'Lugar de nacimiento (país)'")
        return None
    
    # Estadísticas por nacionalidad
    nacionalidades_stats = df.groupby('Lugar de nacimiento (país)').agg({
        'Diferencia_Relativa': ['mean', 'median', 'std', 'count'],
        'Diferencia_Absoluta': ['mean', 'median'],
        'Valor de mercado actual (numérico)': 'mean',
        'Valor_Predicho': 'mean'
    }).round(2)
    
    nacionalidades_stats.columns = ['Dif_Rel_Media', 'Dif_Rel_Mediana', 'Dif_Rel_Std', 'Num_Jugadores',
                                   'Dif_Abs_Media', 'Dif_Abs_Mediana', 'Valor_Real_Promedio', 'Valor_Pred_Promedio']
    
    # Filtrar nacionalidades con al menos 5 jugadores
    nacionalidades_stats = nacionalidades_stats[nacionalidades_stats['Num_Jugadores'] >= 5]
    
    # Ordenar por mayor diferencia relativa absoluta
    nacionalidades_stats['Dif_Rel_Abs'] = abs(nacionalidades_stats['Dif_Rel_Media'])
    nacionalidades_stats = nacionalidades_stats.sort_values('Dif_Rel_Abs', ascending=False)
    
    print("\n--- 🌟 Top 15 nacionalidades con mayor diferencia relativa absoluta ---")
    print(nacionalidades_stats.head(15)[['Dif_Rel_Media', 'Num_Jugadores']])
    
    # Crear visualización con solo un gráfico - Top 10 por diferencia relativa absoluta
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Top 10 nacionalidades por diferencia relativa absoluta
    top_10 = nacionalidades_stats.head(10)
    colors = ['green' if x >= 0 else 'red' for x in top_10['Dif_Rel_Media']]
    
    bars = ax.barh(range(len(top_10)), top_10['Dif_Rel_Media'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index, fontsize=11)
    ax.set_xlabel('Diferencia Relativa Media (%)', fontsize=12)
    ax.set_title('Top 10 Nacionalidades por Mayor Diferencia Relativa', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Calcular límites del eje X para acomodar las etiquetas
    min_val = min(top_10['Dif_Rel_Media'])
    max_val = max(top_10['Dif_Rel_Media'])
    x_range = max_val - min_val
    ax.set_xlim(min_val - x_range * 0.3, max_val + x_range * 0.3)
    
    # Añadir valores en las barras con mejor espaciado
    for i, (bar, value) in enumerate(zip(bars, top_10['Dif_Rel_Media'])):
        ax.text(value + 1.0 if value > 0 else value - 1.0, i, f'{value:.1f}%', 
                va='center', ha='left' if value > 0 else 'right', fontsize=10, fontweight='bold')
    
    # Ajustar márgenes para evitar solapamiento
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    
    # Guardar análisis de nacionalidades
    nacionalidades_viz_file = os.path.join(results_folder, "analisis_nacionalidades.png")
    plt.savefig(nacionalidades_viz_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Análisis de nacionalidades guardado en: {nacionalidades_viz_file}")
    
    # Guardar estadísticas de nacionalidades
    nacionalidades_stats_file = os.path.join(results_folder, "resultados_nacionalidades.csv")
    nacionalidades_stats.to_csv(nacionalidades_stats_file)
    print(f"💾 Estadísticas de nacionalidades guardadas en: {nacionalidades_stats_file}")
    
    return nacionalidades_stats

def sponsor_analysis(df, results_folder):
    """Análisis completo por patrocinadores"""
    print("\n" + "="*80)
    print("👕 ANÁLISIS POR PATROCINADORES")
    print("="*80)
    print("⚠️  NOTA: Análisis basado en datos filtrados (diferencia relativa ≤ 25%)")
    
    if 'Proveedor' not in df.columns:
        print("❌ No se encontró la columna 'Proveedor'")
        return None
    
    # Limpiar datos de proveedor
    df_clean = df[df['Proveedor'].notna() & (df['Proveedor'] != '')]
    
    # Estadísticas por patrocinador
    patrocinadores_stats = df_clean.groupby('Proveedor').agg({
        'Diferencia_Relativa': ['mean', 'median', 'std', 'count'],
        'Diferencia_Absoluta': ['mean', 'median'],
        'Valor de mercado actual (numérico)': 'mean',
        'Valor_Predicho': 'mean'
    }).round(2)
    
    patrocinadores_stats.columns = ['Dif_Rel_Media', 'Dif_Rel_Mediana', 'Dif_Rel_Std', 'Num_Jugadores',
                                   'Dif_Abs_Media', 'Dif_Abs_Mediana', 'Valor_Real_Promedio', 'Valor_Pred_Promedio']
    
    # Filtrar patrocinadores con al menos 3 jugadores
    patrocinadores_stats = patrocinadores_stats[patrocinadores_stats['Num_Jugadores'] >= 3].sort_values('Dif_Rel_Media', ascending=False)
    
    print("\n--- 👕 Patrocinadores con mayor diferencia relativa promedio (datos filtrados) ---")
    print(patrocinadores_stats.head(10))
    
    print("\n--- ⭐ Patrocinadores con menor diferencia relativa promedio (datos filtrados) ---")
    print(patrocinadores_stats.tail(10))
    
    # Crear visualizaciones de patrocinadores (solo 2 gráficos)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Patrocinadores por diferencia relativa
    axes[0].barh(range(len(patrocinadores_stats)), patrocinadores_stats['Dif_Rel_Media'], color='orange')
    axes[0].set_yticks(range(len(patrocinadores_stats)))
    axes[0].set_yticklabels(patrocinadores_stats.index, fontsize=10)
    axes[0].set_xlabel('Diferencia Relativa Media (%)')
    axes[0].set_title('Patrocinadores - Diferencia Relativa Media', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Número de jugadores por patrocinador
    axes[1].bar(range(len(patrocinadores_stats)), patrocinadores_stats['Num_Jugadores'], color='purple')
    axes[1].set_xticks(range(len(patrocinadores_stats)))
    axes[1].set_xticklabels(patrocinadores_stats.index, rotation=45, ha='right', fontsize=10)
    axes[1].set_ylabel('Número de Jugadores')
    axes[1].set_title('Número de Jugadores por Patrocinador', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar análisis de patrocinadores
    patrocinadores_viz_file = os.path.join(results_folder, "analisis_patrocinadores.png")
    plt.savefig(patrocinadores_viz_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Análisis de patrocinadores guardado en: {patrocinadores_viz_file}")
    
    # Guardar estadísticas de patrocinadores
    patrocinadores_stats_file = os.path.join(results_folder, "resultados_patrocinadores.csv")
    patrocinadores_stats.to_csv(patrocinadores_stats_file)
    print(f"💾 Estadísticas de patrocinadores guardadas en: {patrocinadores_stats_file}")
    
    return patrocinadores_stats

def generate_summary_report(df, correlations_df, equipos_stats, nacionalidades_stats, patrocinadores_stats, results_folder):
    """Generar reporte resumen completo"""
    print("\n" + "="*80)
    print("📋 RESUMEN EJECUTIVO")
    print("="*80)
    
    # Crear reporte detallado
    report = f"""
RESUMEN DEL ANÁLISIS EXPLORATORIO DE DATOS - FÚTBOL
==================================================
📋 NOTA IMPORTANTE: Este análisis excluye instancias con diferencia relativa > 25%
🔢 CÁLCULO: Diferencia = Valor Predicho - Valor Real
🟢 Valores positivos: Predicción > Valor Real
🔴 Valores negativos: Predicción < Valor Real

📊 INFORMACIÓN GENERAL:
- Total de jugadores analizados (después del filtro): {df.shape[0]:,}
- Total de variables: {df.shape[1]}
- Equipos únicos: {df['Club actual'].nunique() if 'Club actual' in df.columns else 'N/A'}
- Nacionalidades únicas: {df['Lugar de nacimiento (país)'].nunique() if 'Lugar de nacimiento (país)' in df.columns else 'N/A'}
- Patrocinadores únicos: {df['Proveedor'].nunique() if 'Proveedor' in df.columns else 'N/A'}

🔍 FILTRADO DE DATOS:
- Criterio aplicado: Diferencia relativa absoluta ≤ 25%
- Instancias incluidas en el análisis: {df.shape[0]:,}

💰 DIFERENCIAS VALOR PREDICHO VS VALOR REAL (DATOS FILTRADOS):
- Diferencia promedio: €{df['Diferencia_Absoluta'].mean():,.0f}
- Diferencia relativa promedio: {df['Diferencia_Relativa'].mean():.1f}%
- Desviación estándar: €{df['Diferencia_Absoluta'].std():,.0f}
- Error absoluto medio: €{abs(df['Diferencia_Absoluta']).mean():,.0f}
- Error relativo medio: {abs(df['Diferencia_Relativa']).mean():.2f}%
- 🟢 Predicciones superiores al valor real: {(df['Diferencia_Absoluta'] > 0).sum():,} ({(df['Diferencia_Absoluta'] > 0).mean()*100:.1f}%)
- 🔴 Predicciones inferiores al valor real: {(df['Diferencia_Absoluta'] < 0).sum():,} ({(df['Diferencia_Absoluta'] < 0).mean()*100:.1f}%)

🔗 TOP 5 VARIABLES MÁS CORRELACIONADAS CON VALOR PREDICHO:
"""
    
    # Agregar top correlaciones
    for i, row in correlations_df.head(5).iterrows():
        report += f"   {row['Variable']}: {row['Correlacion']:.4f}\n"
    
    # Agregar información de equipos
    if equipos_stats is not None:
        report += f"\n⚽ EQUIPOS CON MAYOR DIFERENCIA RELATIVA (DATOS FILTRADOS):\n"
        report += f"   (Valores positivos: predicción > valor real)\n"
        for i, (equipo, stats) in enumerate(equipos_stats.head(3).iterrows(), 1):
            report += f"   {i}. {equipo}: {stats['Dif_Rel_Media']:.2f}% ({stats['Num_Jugadores']} jugadores)\n"
    
    # Agregar información de nacionalidades
    if nacionalidades_stats is not None:
        report += f"\n🌍 NACIONALIDADES CON MAYOR DIFERENCIA RELATIVA (DATOS FILTRADOS):\n"
        report += f"   (Valores positivos: predicción > valor real)\n"
        for i, (nacionalidad, stats) in enumerate(nacionalidades_stats.head(3).iterrows(), 1):
            report += f"   {i}. {nacionalidad}: {stats['Dif_Rel_Media']:.2f}% ({stats['Num_Jugadores']} jugadores)\n"
    
    # Agregar información de patrocinadores
    if patrocinadores_stats is not None:
        report += f"\n👕 PATROCINADORES CON MAYOR DIFERENCIA RELATIVA (DATOS FILTRADOS):\n"
        report += f"   (Valores positivos: predicción > valor real)\n"
        for i, (patrocinador, stats) in enumerate(patrocinadores_stats.head(3).iterrows(), 1):
            report += f"   {i}. {patrocinador}: {stats['Dif_Rel_Media']:.2f}% ({stats['Num_Jugadores']} jugadores)\n"
    
    # Agregar lista de archivos generados
    report += f"\n📁 ARCHIVOS GENERADOS EN '{results_folder}':\n"
    report += "   • analisis_eda_general.png\n"
    report += "   • analisis_equipos.png\n"
    report += "   • analisis_nacionalidades.png\n"
    report += "   • analisis_patrocinadores.png\n"
    report += "   • informacion_basica.csv\n"
    report += "   • correlaciones_variables.csv\n"
    report += "   • resultados_equipos.csv\n"
    report += "   • resultados_nacionalidades.csv\n"
    report += "   • resultados_patrocinadores.csv\n"
    report += "   • data_con_diferencias_filtrado.csv (solo instancias con diff. rel. ≤ 25%)\n"
    report += "   • resumen_ejecutivo.txt\n"
    report += f"   • log_analisis_[timestamp].txt\n"
    
    print(report)
    
    # Guardar reporte
    report_file = os.path.join(results_folder, "resumen_ejecutivo.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"💾 Resumen ejecutivo guardado en: {report_file}")
    
    # Guardar dataset filtrado con diferencias
    data_file = os.path.join(results_folder, "data_con_diferencias_filtrado.csv")
    df.to_csv(data_file, index=False)
    print(f"💾 Dataset filtrado con diferencias guardado en: {data_file}")
    print(f"   ⚠️  NOTA: Este archivo contiene solo instancias con diferencia relativa ≤ 25%")
    print(f"   🔢 CÁLCULO: Diferencia = Valor Predicho - Valor Real")

def main():
    """Función principal del análisis"""
    # Crear carpeta de resultados
    results_folder = create_results_folder()
    
    # Configurar logging
    log_file = setup_logging(results_folder)
    
    try:
        # Cargar datos
        file_path = 'Resultados/data.csv'
        df = load_and_clean_data(file_path)
        
        # Análisis exploratorio básico
        numeric_cols, categorical_cols = basic_eda(df, results_folder)
        
        # Calcular diferencias
        df = calculate_differences(df)
        
        # Análisis de correlaciones
        correlations_df = correlation_analysis(df, results_folder)
        
        # Crear visualizaciones principales
        create_visualizations(df, correlations_df, results_folder)
        
        # Análisis por grupos
        equipos_stats = team_analysis(df, results_folder)
        nacionalidades_stats = nationality_analysis(df, results_folder)
        patrocinadores_stats = sponsor_analysis(df, results_folder)
        
        # Generar reporte final
        generate_summary_report(df, correlations_df, equipos_stats, nacionalidades_stats, 
                               patrocinadores_stats, results_folder)
        
        print(f"\n🎉 ¡ANÁLISIS COMPLETADO CON ÉXITO!")
        print(f"📁 Todos los archivos guardados en: {results_folder}")
        print(f"📝 Log detallado guardado en: {log_file}")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 