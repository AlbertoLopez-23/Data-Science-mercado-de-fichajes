import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import os
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

class TeeOutput:
    """Clase para duplicar la salida a consola y archivo"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def crear_carpeta_kmeans():
    """Crear carpeta k-means si no existe"""
    os.makedirs('k-means', exist_ok=True)
    print("📁 Carpeta 'k-means' creada/verificada")

def crear_boxplots_detallados(df_etiquetado, k_final, nombre_archivo, columna_objetivo):
    """
    Crear boxplots detallados para cada cluster mostrando la distribución del valor de mercado
    """
    print(f"\n📊 CREANDO BOXPLOTS DETALLADOS para {nombre_archivo}")
    
    # Filtrar solo filas con cluster asignado (no -1)
    df_con_cluster = df_etiquetado[df_etiquetado['Cluster'] >= 0].copy()
    
    if len(df_con_cluster) == 0:
        print("   ❌ No hay datos con clusters asignados para crear boxplots")
        return
    
    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Análisis de Clusters - {nombre_archivo.replace(".csv", "").replace("08_db_", "").title()}', 
                 fontsize=16, fontweight='bold')
    
    # Subplot 1: Boxplot principal de valor de mercado por cluster
    ax1 = axes[0, 0]
    
    # Preparar datos para boxplot
    cluster_data = []
    cluster_labels = []
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, k_final))
    
    for i in range(k_final):
        cluster_valores = df_con_cluster[df_con_cluster['Cluster'] == i][columna_objetivo]
        if len(cluster_valores) > 0:
            cluster_data.append(cluster_valores)
            cluster_labels.append(f'Cluster {i}\n(n={len(cluster_valores)})')
    
    # Crear boxplot
    bp1 = ax1.boxplot(cluster_data, labels=cluster_labels, patch_artist=True, 
                      showmeans=True, meanline=True)
    
    # Colorear boxplots
    for patch, color in zip(bp1['boxes'], cluster_colors[:len(cluster_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Distribución de Valor de Mercado por Cluster')
    ax1.set_ylabel('Valor de Mercado ($)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Formatear eje Y con notación de millones
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Subplot 2: Histograma de distribución por cluster
    ax2 = axes[0, 1]
    
    for i in range(k_final):
        cluster_valores = df_con_cluster[df_con_cluster['Cluster'] == i][columna_objetivo]
        if len(cluster_valores) > 0:
            ax2.hist(cluster_valores, alpha=0.6, label=f'Cluster {i}', 
                    color=cluster_colors[i], bins=15, density=True)
    
    ax2.set_title('Distribución de Densidad por Cluster')
    ax2.set_xlabel('Valor de Mercado ($)')
    ax2.set_ylabel('Densidad')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Subplot 3: Violin plot
    ax3 = axes[1, 0]
    
    # Preparar datos para violin plot
    violin_data = []
    violin_positions = []
    for i in range(k_final):
        cluster_valores = df_con_cluster[df_con_cluster['Cluster'] == i][columna_objetivo]
        if len(cluster_valores) > 0:
            violin_data.append(cluster_valores)
            violin_positions.append(i)
    
    if violin_data:
        parts = ax3.violinplot(violin_data, positions=violin_positions, showmeans=True, showmedians=True)
        
        # Colorear violin plots
        for pc, color in zip(parts['bodies'], cluster_colors[:len(violin_data)]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax3.set_title('Distribución Detallada (Violin Plot)')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Valor de Mercado ($)')
    ax3.set_xticks(range(k_final))
    ax3.set_xticklabels([f'C{i}' for i in range(k_final)])
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Subplot 4: Estadísticas por cluster
    ax4 = axes[1, 1]
    ax4.axis('off')  # Quitar ejes para mostrar tabla
    
    # Crear tabla de estadísticas
    stats_data = []
    for i in range(k_final):
        cluster_valores = df_con_cluster[df_con_cluster['Cluster'] == i][columna_objetivo]
        if len(cluster_valores) > 0:
            stats_data.append([
                f'Cluster {i}',
                len(cluster_valores),
                f'${cluster_valores.mean():,.0f}',
                f'${cluster_valores.median():,.0f}',
                f'${cluster_valores.std():,.0f}',
                f'${cluster_valores.min():,.0f}',
                f'${cluster_valores.max():,.0f}'
            ])
    
    if stats_data:
        headers = ['Cluster', 'N', 'Media', 'Mediana', 'Std', 'Min', 'Max']
        table = ax4.table(cellText=stats_data, colLabels=headers, 
                         cellLoc='center', loc='center',
                         colColours=['lightblue']*len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.set_title('Estadísticas por Cluster', pad=20)
    
    plt.tight_layout()
    
    # Guardar gráfico
    nombre_limpio = nombre_archivo.replace('.csv', '')
    plt.savefig(f'k-means/05_boxplots_detallados_{nombre_limpio}_k{k_final}.png', 
                dpi=300, bbox_inches='tight')
    print(f"📊 Boxplots detallados guardados: k-means/05_boxplots_detallados_{nombre_limpio}_k{k_final}.png")
    plt.close()
    
    # Crear gráfico adicional: Comparación de clusters
    crear_grafico_comparacion_clusters(df_con_cluster, k_final, nombre_archivo, columna_objetivo)

def crear_grafico_comparacion_clusters(df_con_cluster, k_final, nombre_archivo, columna_objetivo):
    """
    Crear gráfico de comparación entre clusters con múltiples métricas
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Comparación de Clusters - {nombre_archivo.replace(".csv", "").replace("08_db_", "").title()}', 
                 fontsize=14, fontweight='bold')
    
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, k_final))
    
    # Gráfico 1: Tamaño de clusters
    ax1 = axes[0]
    cluster_sizes = [len(df_con_cluster[df_con_cluster['Cluster'] == i]) for i in range(k_final)]
    bars1 = ax1.bar(range(k_final), cluster_sizes, color=cluster_colors, alpha=0.7)
    ax1.set_title('Tamaño de Clusters')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Número de Jugadores')
    ax1.set_xticks(range(k_final))
    ax1.set_xticklabels([f'C{i}' for i in range(k_final)])
    
    # Agregar valores en las barras
    for bar, size in zip(bars1, cluster_sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(cluster_sizes)*0.01,
                f'{size}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Valor promedio por cluster
    ax2 = axes[1]
    cluster_means = [df_con_cluster[df_con_cluster['Cluster'] == i][columna_objetivo].mean() 
                     for i in range(k_final)]
    bars2 = ax2.bar(range(k_final), cluster_means, color=cluster_colors, alpha=0.7)
    ax2.set_title('Valor Promedio por Cluster')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Valor Promedio ($)')
    ax2.set_xticks(range(k_final))
    ax2.set_xticklabels([f'C{i}' for i in range(k_final)])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Agregar valores en las barras
    for bar, mean_val in zip(bars2, cluster_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(cluster_means)*0.01,
                f'${mean_val/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Rango de valores (min-max) por cluster
    ax3 = axes[2]
    cluster_mins = [df_con_cluster[df_con_cluster['Cluster'] == i][columna_objetivo].min() 
                    for i in range(k_final)]
    cluster_maxs = [df_con_cluster[df_con_cluster['Cluster'] == i][columna_objetivo].max() 
                    for i in range(k_final)]
    cluster_ranges = [max_val - min_val for min_val, max_val in zip(cluster_mins, cluster_maxs)]
    
    bars3 = ax3.bar(range(k_final), cluster_ranges, color=cluster_colors, alpha=0.7)
    ax3.set_title('Rango de Valores por Cluster')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Rango (Max - Min) ($)')
    ax3.set_xticks(range(k_final))
    ax3.set_xticklabels([f'C{i}' for i in range(k_final)])
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Agregar valores en las barras
    for bar, range_val in zip(bars3, cluster_ranges):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(cluster_ranges)*0.01,
                f'${range_val/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfico
    nombre_limpio = nombre_archivo.replace('.csv', '')
    plt.savefig(f'k-means/06_comparacion_clusters_{nombre_limpio}_k{k_final}.png', 
                dpi=300, bbox_inches='tight')
    print(f"📊 Comparación de clusters guardada: k-means/06_comparacion_clusters_{nombre_limpio}_k{k_final}.png")
    plt.close()

def metodo_del_codo(X, nombre_archivo, max_k=10):
    """
    Implementa el método del codo para encontrar el número óptimo de clusters
    """
    print(f"\n🔍 MÉTODO DEL CODO para {nombre_archivo}")
    print("-" * 50)
    
    # Calcular inercia para diferentes valores de K
    inercias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
        print(f"   K={k}: Inercia = {kmeans.inertia_:.2f}")
    
    # Crear gráfico del codo
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inercias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia (WCSS)')
    plt.title(f'Método del Codo - {nombre_archivo}')
    plt.grid(True, alpha=0.3)
    
    # Método mejorado para encontrar el codo usando el método de la distancia perpendicular
    if len(inercias) >= 3:
        # Normalizar los datos para el cálculo
        x_norm = np.array(K_range)
        y_norm = np.array(inercias)
        
        # Calcular distancias perpendiculares a la línea que conecta primer y último punto
        x1, y1 = x_norm[0], y_norm[0]  # Primer punto
        x2, y2 = x_norm[-1], y_norm[-1]  # Último punto
        
        # Calcular distancias perpendiculares para cada punto
        distances = []
        for i in range(len(x_norm)):
            x_point, y_point = x_norm[i], y_norm[i]
            # Fórmula de distancia perpendicular de un punto a una línea
            numerator = abs((y2 - y1) * x_point - (x2 - x1) * y_point + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distance = numerator / denominator
            distances.append(distance)
        
        # El codo está en el punto con mayor distancia perpendicular
        codo_idx = np.argmax(distances)
        k_optimo = K_range[codo_idx]
        
        # Método alternativo: buscar el mayor cambio en la pendiente
        if len(inercias) >= 4:
            # Calcular diferencias y segunda derivada
            diff1 = np.diff(inercias)
            diff2 = np.diff(diff1)
            
            # Encontrar el punto donde la segunda derivada es máxima (cambio más pronunciado)
            if len(diff2) > 0:
                codo_idx_alt = np.argmax(diff2) + 2  # +2 porque perdemos elementos en las diferencias
                k_optimo_alt = K_range[codo_idx_alt] if codo_idx_alt < len(K_range) else k_optimo
                
                # Si los métodos difieren mucho, usar el más conservador (menor K)
                if abs(k_optimo - k_optimo_alt) > 1:
                    k_optimo = min(k_optimo, k_optimo_alt)
                    print(f"   Métodos del codo difieren: usando K más conservador = {k_optimo}")
        
        plt.axvline(x=k_optimo, color='red', linestyle='--', alpha=0.7, 
                   label=f'K óptimo sugerido: {k_optimo}')
        plt.legend()
        
        print(f"   🎯 K óptimo detectado por método del codo: {k_optimo}")
        print(f"   📊 Distancia perpendicular máxima en K={k_optimo}: {distances[codo_idx]:.2f}")
    else:
        k_optimo = 3  # Valor por defecto
        print(f"   ⚠️ Pocos puntos para análisis, usando K por defecto: {k_optimo}")
    
    # Guardar gráfico
    nombre_limpio = nombre_archivo.replace('.csv', '')
    plt.savefig(f'k-means/01_metodo_codo_{nombre_limpio}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico del codo guardado: k-means/01_metodo_codo_{nombre_limpio}.png")
    plt.close()
    
    return inercias, k_optimo

def analisis_silueta(X, nombre_archivo, k_range=range(2, 8)):
    """
    Análisis de silueta para diferentes valores de K
    """
    print(f"\n📊 ANÁLISIS DE SILUETA para {nombre_archivo}")
    print("-" * 50)
    
    scores_silueta = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        scores_silueta.append(score)
        print(f"   K={k}: Score de silueta = {score:.4f}")
    
    # Crear gráfico de silueta
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores_silueta, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Score de Silueta')
    plt.title(f'Análisis de Silueta - {nombre_archivo}')
    plt.grid(True, alpha=0.3)
    
    # Marcar el mejor K
    mejor_k = k_range[np.argmax(scores_silueta)]
    mejor_score = max(scores_silueta)
    plt.axvline(x=mejor_k, color='red', linestyle='--', alpha=0.7,
               label=f'Mejor K: {mejor_k} (Score: {mejor_score:.4f})')
    plt.legend()
    
    # Guardar gráfico
    nombre_limpio = nombre_archivo.replace('.csv', '')
    plt.savefig(f'k-means/02_analisis_silueta_{nombre_limpio}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico de silueta guardado: k-means/02_analisis_silueta_{nombre_limpio}.png")
    plt.close()
    
    return scores_silueta, mejor_k, mejor_score

def grafico_silueta_detallado(X, labels, k, nombre_archivo):
    """
    Crear gráfico detallado de silueta para un K específico
    """
    # Calcular scores de silueta
    sample_silhouette_values = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    y_lower = 10
    for i in range(k):
        # Valores de silueta para cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        # Etiquetar clusters
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_xlabel('Valores de Coeficiente de Silueta')
    ax.set_ylabel('Índice de Cluster')
    ax.set_title(f'Gráfico de Silueta para K={k} - {nombre_archivo}\n'
                f'Score promedio: {silhouette_avg:.4f}')
    
    # Línea vertical para el score promedio
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
              label=f'Score promedio: {silhouette_avg:.4f}')
    ax.legend()
    
    # Guardar gráfico
    nombre_limpio = nombre_archivo.replace('.csv', '')
    plt.savefig(f'k-means/03_silueta_detallada_{nombre_limpio}_k{k}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico detallado de silueta guardado: k-means/03_silueta_detallada_{nombre_limpio}_k{k}.png")
    plt.close()
    
    return silhouette_avg

def visualizar_clusters_2d(X_original, X_scaled, labels, centroids, k, nombre_archivo, feature_names, columna_objetivo):
    """
    Visualizar clusters en 2D usando PCA
    """
    # Aplicar PCA para reducir a 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(centroids)
    
    # Crear figura con subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Clusters en espacio PCA
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, k))
    for i in range(k):
        mask = labels == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
    
    # Centroids en PCA
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    ax1.set_title(f'Clusters en Espacio PCA (K={k})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Encontrar la columna del valor de mercado en X_original
    if isinstance(X_original, pd.DataFrame):
        if columna_objetivo in X_original.columns:
            valor_mercado = X_original[columna_objetivo].values
        else:
            # Si no está, usar la primera columna numérica
            valor_mercado = X_original.iloc[:, 0].values
    else:
        # Si es numpy array, usar la primera columna
        valor_mercado = X_original[:, 0]
    
    # Gráfico 2: Distribución del valor de mercado por cluster
    for i in range(k):
        mask = labels == i
        ax2.hist(valor_mercado[mask], alpha=0.7, label=f'Cluster {i}', 
                bins=20, color=colors[i])
    ax2.set_xlabel('Valor de Mercado')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Valor de Mercado por Cluster')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Boxplot del valor de mercado por cluster
    data_boxplot = [valor_mercado[labels == i] for i in range(k)]
    bp = ax3.boxplot(data_boxplot, labels=[f'C{i}' for i in range(k)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Valor de Mercado')
    ax3.set_title('Distribución de Valor de Mercado por Cluster')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Tamaño de clusters
    cluster_sizes = [np.sum(labels == i) for i in range(k)]
    ax4.bar(range(k), cluster_sizes, color=colors, alpha=0.7)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Número de Jugadores')
    ax4.set_title('Tamaño de Clusters')
    ax4.set_xticks(range(k))
    ax4.set_xticklabels([f'C{i}' for i in range(k)])
    for i, size in enumerate(cluster_sizes):
        ax4.text(i, size + max(cluster_sizes)*0.01, str(size), 
                ha='center', va='bottom')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfico
    nombre_limpio = nombre_archivo.replace('.csv', '')
    plt.savefig(f'k-means/04_visualizacion_clusters_{nombre_limpio}_k{k}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Visualización de clusters guardada: k-means/04_visualizacion_clusters_{nombre_limpio}_k{k}.png")
    plt.close()
    
    return pca.explained_variance_ratio_

def crear_grafico_dispersion_clusters(df_etiquetado, k_final, nombre_archivo, columna_objetivo, columna_overall):
    """
    Crear gráfico de dispersión coloreado por clusters usando valor de mercado y overall
    """
    print(f"\n🎨 CREANDO GRÁFICO DE DISPERSIÓN POR CLUSTERS para {nombre_archivo}")
    
    # Filtrar solo filas con cluster asignado (no -1)
    df_con_cluster = df_etiquetado[df_etiquetado['Cluster'] >= 0].copy()
    
    if len(df_con_cluster) == 0:
        print("   ❌ No hay datos con clusters asignados para crear dispersión")
        return
    
    # Crear figura principal
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Análisis de Dispersión por Clusters - {nombre_archivo.replace(".csv", "").replace("08_db_", "").title()}', 
                 fontsize=16, fontweight='bold')
    
    # Definir colores para los clusters
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, k_final))
    
    # Gráfico 1: Dispersión principal (Overall vs Valor de Mercado)
    ax1 = axes[0, 0]
    
    for i in range(k_final):
        cluster_data = df_con_cluster[df_con_cluster['Cluster'] == i]
        if len(cluster_data) > 0:
            ax1.scatter(cluster_data[columna_overall], cluster_data[columna_objetivo], 
                       c=[cluster_colors[i]], label=f'Cluster {i} (n={len(cluster_data)})', 
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Overall Rating')
    ax1.set_ylabel('Valor de Mercado ($)')
    ax1.set_title('Dispersión: Overall Rating vs Valor de Mercado')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Gráfico 2: Dispersión con densidad
    ax2 = axes[0, 1]
    
    # Crear un scatter plot con diferentes tamaños basados en densidad local
    for i in range(k_final):
        cluster_data = df_con_cluster[df_con_cluster['Cluster'] == i]
        if len(cluster_data) > 0:
            # Calcular tamaños basados en la densidad del cluster
            sizes = np.full(len(cluster_data), 50 + 100/max(1, len(cluster_data)/10))
            ax2.scatter(cluster_data[columna_overall], cluster_data[columna_objetivo], 
                       c=[cluster_colors[i]], label=f'Cluster {i}', 
                       alpha=0.6, s=sizes, edgecolors='white', linewidth=1)
    
    ax2.set_xlabel('Overall Rating')
    ax2.set_ylabel('Valor de Mercado ($)')
    ax2.set_title('Dispersión con Densidad por Cluster')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Gráfico 3: Distribución marginal de Overall Rating
    ax3 = axes[1, 0]
    
    for i in range(k_final):
        cluster_data = df_con_cluster[df_con_cluster['Cluster'] == i]
        if len(cluster_data) > 0:
            ax3.hist(cluster_data[columna_overall], alpha=0.6, label=f'Cluster {i}', 
                    color=cluster_colors[i], bins=15, density=True)
    
    ax3.set_xlabel('Overall Rating')
    ax3.set_ylabel('Densidad')
    ax3.set_title('Distribución de Overall Rating por Cluster')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Estadísticas comparativas
    ax4 = axes[1, 1]
    
    # Crear gráfico de barras comparativo
    cluster_stats = []
    cluster_labels = []
    overall_means = []
    value_means = []
    
    for i in range(k_final):
        cluster_data = df_con_cluster[df_con_cluster['Cluster'] == i]
        if len(cluster_data) > 0:
            cluster_labels.append(f'C{i}')
            overall_means.append(cluster_data[columna_overall].mean())
            value_means.append(cluster_data[columna_objetivo].mean() / 1e6)  # En millones
    
    x = np.arange(len(cluster_labels))
    width = 0.35
    
    # Normalizar overall ratings para comparación visual
    overall_normalized = np.array(overall_means) / max(overall_means) * max(value_means)
    
    bars1 = ax4.bar(x - width/2, value_means, width, label='Valor Medio (M$)', 
                    color=[cluster_colors[i] for i in range(len(cluster_labels))], alpha=0.7)
    bars2 = ax4.bar(x + width/2, overall_normalized, width, label='Overall Medio (Norm.)', 
                    color=[cluster_colors[i] for i in range(len(cluster_labels))], alpha=0.4)
    
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Valor')
    ax4.set_title('Comparación de Medias por Cluster')
    ax4.set_xticks(x)
    ax4.set_xticklabels(cluster_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, val in zip(bars1, value_means):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(value_means)*0.01,
                f'{val:.1f}M', ha='center', va='bottom', fontsize=9)
    
    for bar, val in zip(bars2, overall_means):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(overall_normalized)*0.01,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar gráfico
    nombre_limpio = nombre_archivo.replace('.csv', '')
    plt.savefig(f'k-means/07_dispersion_clusters_{nombre_limpio}_k{k_final}.png', 
                dpi=300, bbox_inches='tight')
    print(f"🎨 Gráfico de dispersión guardado: k-means/07_dispersion_clusters_{nombre_limpio}_k{k_final}.png")
    plt.close()
    
    # Crear gráfico adicional de dispersión grande
    crear_grafico_dispersion_grande(df_con_cluster, k_final, nombre_archivo, columna_objetivo, columna_overall, cluster_colors)

def crear_grafico_dispersion_grande(df_con_cluster, k_final, nombre_archivo, columna_objetivo, columna_overall, cluster_colors):
    """
    Crear un gráfico de dispersión grande y detallado
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Crear scatter plot principal
    for i in range(k_final):
        cluster_data = df_con_cluster[df_con_cluster['Cluster'] == i]
        if len(cluster_data) > 0:
            ax.scatter(cluster_data[columna_overall], cluster_data[columna_objetivo], 
                      c=[cluster_colors[i]], label=f'Cluster {i} (n={len(cluster_data)})', 
                      alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    # Calcular y dibujar centroides
    centroids_overall = []
    centroids_value = []
    
    for i in range(k_final):
        cluster_data = df_con_cluster[df_con_cluster['Cluster'] == i]
        if len(cluster_data) > 0:
            centroid_overall = cluster_data[columna_overall].mean()
            centroid_value = cluster_data[columna_objetivo].mean()
            centroids_overall.append(centroid_overall)
            centroids_value.append(centroid_value)
            
            # Dibujar centroide
            ax.scatter(centroid_overall, centroid_value, 
                      c='red', marker='X', s=200, linewidths=2, 
                      edgecolors='black', alpha=0.9)
            
            # Etiqueta del centroide
            ax.annotate(f'C{i}', (centroid_overall, centroid_value), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=12, fontweight='bold', color='red')
    
    ax.set_xlabel('Overall Rating', fontsize=14)
    ax.set_ylabel('Valor de Mercado ($)', fontsize=14)
    ax.set_title(f'Clustering: Overall Rating vs Valor de Mercado\n{nombre_archivo.replace(".csv", "").replace("08_db_", "").title()}', 
                fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Añadir estadísticas en el gráfico
    textstr = f'Clusters: {k_final}\n'
    textstr += f'Total jugadores: {len(df_con_cluster)}\n'
    textstr += f'Overall range: {df_con_cluster[columna_overall].min():.0f}-{df_con_cluster[columna_overall].max():.0f}\n'
    textstr += f'Valor range: ${df_con_cluster[columna_objetivo].min()/1e6:.1f}M-${df_con_cluster[columna_objetivo].max()/1e6:.1f}M'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Guardar gráfico
    nombre_limpio = nombre_archivo.replace('.csv', '')
    plt.savefig(f'k-means/08_dispersion_grande_{nombre_limpio}_k{k_final}.png', 
                dpi=300, bbox_inches='tight')
    print(f"🎨 Gráfico de dispersión grande guardado: k-means/08_dispersion_grande_{nombre_limpio}_k{k_final}.png")
    plt.close()

def evaluar_calidad_clusters(X_scaled, labels, k_final, nombre_archivo, kmeans_model):
    """
    Evaluar la calidad de los clusters usando múltiples métricas
    """
    print(f"\n📊 EVALUACIÓN DE CALIDAD DE CLUSTERS:")
    print("-" * 50)
    
    # 1. Score de Silueta
    silueta_score = silhouette_score(X_scaled, labels)
    print(f"   🎯 Score de Silueta Global: {silueta_score:.4f}")
    
    # Interpretación del score de silueta
    if silueta_score >= 0.7:
        calidad_silueta = "EXCELENTE"
        emoji_silueta = "🟢"
    elif silueta_score >= 0.5:
        calidad_silueta = "BUENA"
        emoji_silueta = "🟡"
    elif silueta_score >= 0.25:
        calidad_silueta = "REGULAR"
        emoji_silueta = "🟠"
    else:
        calidad_silueta = "POBRE"
        emoji_silueta = "🔴"
    
    print(f"   {emoji_silueta} Calidad según Silueta: {calidad_silueta}")
    
    # 2. Inercia y métricas relacionadas
    inercia_total = kmeans_model.inertia_
    print(f"   📉 Inercia Total (WCSS): {inercia_total:.2f}")
    
    # Calcular inercia promedio por cluster
    inercia_promedio = inercia_total / k_final
    print(f"   📊 Inercia Promedio por Cluster: {inercia_promedio:.2f}")
    
    # 3. Análisis por cluster individual
    print(f"\n   📋 ANÁLISIS POR CLUSTER INDIVIDUAL:")
    
    # Calcular scores de silueta por muestra
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    
    cluster_qualities = []
    for i in range(k_final):
        # Filtrar datos de este cluster
        cluster_mask = labels == i
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size > 0:
            # Score de silueta para este cluster
            cluster_silhouette_values = sample_silhouette_values[cluster_mask]
            cluster_silhouette_mean = cluster_silhouette_values.mean()
            cluster_silhouette_std = cluster_silhouette_values.std()
            
            # Calcular inercia intra-cluster (distancia al centroide)
            cluster_data = X_scaled[cluster_mask]
            centroid = kmeans_model.cluster_centers_[i]
            intra_cluster_distances = np.sum((cluster_data - centroid) ** 2, axis=1)
            inercia_cluster = np.sum(intra_cluster_distances)
            inercia_promedio_cluster = inercia_cluster / cluster_size
            
            print(f"     Cluster {i}:")
            print(f"       👥 Tamaño: {cluster_size}")
            print(f"       🎯 Silueta promedio: {cluster_silhouette_mean:.4f} ± {cluster_silhouette_std:.4f}")
            print(f"       📉 Inercia intra-cluster: {inercia_cluster:.2f}")
            print(f"       📊 Inercia promedio por punto: {inercia_promedio_cluster:.4f}")
            
            # Evaluar calidad individual del cluster
            if cluster_silhouette_mean >= 0.5:
                calidad_individual = "BUENA"
                emoji_individual = "🟢"
            elif cluster_silhouette_mean >= 0.25:
                calidad_individual = "REGULAR"
                emoji_individual = "🟡"
            else:
                calidad_individual = "POBRE"
                emoji_individual = "🔴"
            
            print(f"       {emoji_individual} Calidad: {calidad_individual}")
            
            cluster_qualities.append({
                'cluster': i,
                'size': cluster_size,
                'silhouette_mean': cluster_silhouette_mean,
                'silhouette_std': cluster_silhouette_std,
                'inertia': inercia_cluster,
                'inertia_per_point': inercia_promedio_cluster,
                'quality': calidad_individual
            })
    
    # 4. Métricas de cohesión y separación
    print(f"\n   📈 MÉTRICAS DE COHESIÓN Y SEPARACIÓN:")
    
    # Distancia entre centroides (separación)
    centroids = kmeans_model.cluster_centers_
    min_centroid_distance = float('inf')
    max_centroid_distance = 0
    
    for i in range(k_final):
        for j in range(i+1, k_final):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            min_centroid_distance = min(min_centroid_distance, distance)
            max_centroid_distance = max(max_centroid_distance, distance)
    
    print(f"     🔄 Distancia mínima entre centroides: {min_centroid_distance:.4f}")
    print(f"     🔄 Distancia máxima entre centroides: {max_centroid_distance:.4f}")
    print(f"     📏 Ratio separación (max/min): {max_centroid_distance/min_centroid_distance:.2f}")
    
    # 5. Resumen de calidad
    print(f"\n   🏆 RESUMEN DE CALIDAD:")
    clusters_buenos = sum(1 for c in cluster_qualities if c['quality'] == 'BUENA')
    clusters_regulares = sum(1 for c in cluster_qualities if c['quality'] == 'REGULAR')
    clusters_pobres = sum(1 for c in cluster_qualities if c['quality'] == 'POBRE')
    
    print(f"     🟢 Clusters de calidad BUENA: {clusters_buenos}/{k_final}")
    print(f"     🟡 Clusters de calidad REGULAR: {clusters_regulares}/{k_final}")
    print(f"     🔴 Clusters de calidad POBRE: {clusters_pobres}/{k_final}")
    
    # Calidad general del clustering
    if clusters_pobres == 0 and clusters_buenos >= k_final // 2:
        calidad_general = "EXCELENTE"
        emoji_general = "🌟"
    elif clusters_pobres <= k_final // 4:
        calidad_general = "BUENA"
        emoji_general = "✅"
    elif clusters_pobres <= k_final // 2:
        calidad_general = "REGULAR"
        emoji_general = "⚠️"
    else:
        calidad_general = "POBRE"
        emoji_general = "❌"
    
    print(f"     {emoji_general} CALIDAD GENERAL DEL CLUSTERING: {calidad_general}")
    
    return {
        'silhouette_score': silueta_score,
        'inertia_total': inercia_total,
        'inertia_promedio': inercia_promedio,
        'min_centroid_distance': min_centroid_distance,
        'max_centroid_distance': max_centroid_distance,
        'separation_ratio': max_centroid_distance/min_centroid_distance,
        'clusters_buenos': clusters_buenos,
        'clusters_regulares': clusters_regulares,
        'clusters_pobres': clusters_pobres,
        'calidad_general': calidad_general,
        'cluster_details': cluster_qualities
    }

def procesar_archivo_csv(archivo_path, carpeta_origen="DB_separadas"):
    """
    Procesar un archivo CSV individual para clustering usando solo Overall Rating y Valor de Mercado
    """
    nombre_archivo = os.path.basename(archivo_path)
    print(f"\n{'='*60}")
    print(f"🎯 PROCESANDO: {nombre_archivo}")
    print(f"{'='*60}")
    
    try:
        # Cargar datos
        df = pd.read_csv(archivo_path)
        print(f"📊 Dimensiones del dataset: {df.shape}")
        
        # Definir las dos columnas específicas para clustering
        columna_objetivo = 'Valor de mercado actual (numérico)'
        columna_overall = 'overallrating'
        
        # Verificar que existen ambas columnas
        if columna_objetivo not in df.columns:
            print(f"❌ ERROR: Columna '{columna_objetivo}' no encontrada")
            return None
            
        if columna_overall not in df.columns:
            print(f"❌ ERROR: Columna '{columna_overall}' no encontrada")
            return None
        
        # Preparar datos para clustering
        print(f"\n🔧 PREPARANDO DATOS PARA CLUSTERING:")
        print(f"   📊 Variables seleccionadas para clustering:")
        print(f"   - {columna_objetivo} (Variable objetivo)")
        print(f"   - {columna_overall} (Rating FIFA)")
        
        # Crear dataset para clustering con solo estas dos variables
        X_original = df[[columna_objetivo, columna_overall]].copy()
        
        # Eliminar filas con valores faltantes
        filas_antes = len(X_original)
        X_original = X_original.dropna()
        filas_despues = len(X_original)
        
        print(f"   Filas antes de limpiar: {filas_antes}")
        print(f"   Filas después de limpiar: {filas_despues}")
        print(f"   Filas eliminadas: {filas_antes - filas_despues}")
        
        if len(X_original) < 10:
            print(f"❌ ERROR: Muy pocas muestras válidas ({len(X_original)})")
            return None
        
        # Estadísticas de las variables
        print(f"\n📈 ESTADÍSTICAS DE LAS VARIABLES:")
        print(f"   {columna_objetivo}:")
        print(f"     Mínimo: ${X_original[columna_objetivo].min():,.0f}")
        print(f"     Máximo: ${X_original[columna_objetivo].max():,.0f}")
        print(f"     Promedio: ${X_original[columna_objetivo].mean():,.0f}")
        print(f"     Mediana: ${X_original[columna_objetivo].median():,.0f}")
        
        print(f"   {columna_overall}:")
        print(f"     Mínimo: {X_original[columna_overall].min():.0f}")
        print(f"     Máximo: {X_original[columna_overall].max():.0f}")
        print(f"     Promedio: {X_original[columna_overall].mean():.1f}")
        print(f"     Mediana: {X_original[columna_overall].median():.0f}")
        
        # Estandarizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_original)
        
        print(f"   Datos estandarizados: {X_scaled.shape}")
        
        # Determinar rango de K para análisis
        max_k = min(10, len(X_original) // 5)  # Máximo 10 o 1/5 del dataset
        k_range = range(2, max_k + 1)
        
        print(f"   Rango de K para análisis: {list(k_range)}")
        
        # 1. Método del codo
        inercias, k_codo = metodo_del_codo(X_scaled, nombre_archivo, max_k)
        
        # 2. Análisis de silueta
        scores_silueta, k_silueta, mejor_score_silueta = analisis_silueta(X_scaled, nombre_archivo, k_range)
        
        # 3. Decidir K final - PRIORIDAD AL MÉTODO DEL CODO
        k_final = k_codo  # Siempre usar el método del codo como primera opción
        
        # Solo ajustar si el codo sugiere algo extremo
        if k_codo < 2:
            k_final = 2  # Mínimo 2 clusters
        elif k_codo > max_k:
            k_final = max_k  # No exceder el máximo
        
        print(f"\n🎯 DECISIÓN DE K:")
        print(f"   K sugerido por método del codo: {k_codo} ⭐ (PRIORITARIO)")
        print(f"   K sugerido por análisis de silueta: {k_silueta} (score: {mejor_score_silueta:.4f})")
        print(f"   K FINAL seleccionado: {k_final}")
        print(f"   Lógica aplicada: Prioridad absoluta al método del codo")
        
        # Mostrar comparación informativa
        if k_codo == k_silueta:
            print(f"   ✅ Ambos métodos coinciden en K={k_final}")
        elif abs(k_codo - k_silueta) <= 1:
            print(f"   ✅ Métodos muy cercanos (diferencia ≤ 1)")
        else:
            print(f"   ⚠️ Métodos difieren significativamente (diferencia = {abs(k_codo - k_silueta)})")
            print(f"       Manteniendo K del codo = {k_final} por prioridad establecida")
        
        # 4. Aplicar K-means con K final
        print(f"\n🔄 APLICANDO K-MEANS CON K={k_final}:")
        
        kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
        labels_final = kmeans_final.fit_predict(X_scaled)
        centroids_final = kmeans_final.cluster_centers_
        
        # ASEGURAR QUE LOS CLUSTERS SEAN SOLO POSITIVOS (0, 1, 2, ...)
        # Los clusters ya son positivos por defecto en sklearn, pero verificamos
        assert np.all(labels_final >= 0), "Error: Se encontraron clusters negativos"
        
        # Métricas básicas
        inercia_final = kmeans_final.inertia_
        silueta_final = silhouette_score(X_scaled, labels_final)
        
        print(f"   ✅ Clustering completado exitosamente")
        print(f"   📊 Clusters generados: {sorted(np.unique(labels_final))}")
        print(f"   📉 Inercia total: {inercia_final:.2f}")
        print(f"   🎯 Score de silueta: {silueta_final:.4f}")
        
        # 5. EVALUACIÓN DETALLADA DE CALIDAD
        calidad_info = evaluar_calidad_clusters(X_scaled, labels_final, k_final, nombre_archivo, kmeans_final)
        
        # 6. Análisis de clusters con información de calidad
        print(f"\n📊 ANÁLISIS DE CLUSTERS CON DATOS ORIGINALES:")
        for i in range(k_final):
            mask = labels_final == i
            cluster_size = np.sum(mask)
            
            # Obtener valores para este cluster
            cluster_valores = X_original.iloc[mask][columna_objetivo]
            cluster_overall = X_original.iloc[mask][columna_overall]
            
            # Obtener información de calidad de este cluster
            cluster_quality_info = next(
                (c for c in calidad_info['cluster_details'] if c['cluster'] == i), 
                None
            )
            
            print(f"   Cluster {i}: {cluster_size} jugadores")
            print(f"     💰 Valor promedio: ${cluster_valores.mean():,.0f}")
            print(f"     🎮 Overall promedio: {cluster_overall.mean():.1f}")
            print(f"     💰 Valor mediana: ${cluster_valores.median():,.0f}")
            print(f"     🎮 Overall mediana: {cluster_overall.median():.0f}")
            
            if cluster_quality_info:
                emoji_calidad = "🟢" if cluster_quality_info['quality'] == 'BUENA' else "🟡" if cluster_quality_info['quality'] == 'REGULAR' else "🔴"
                print(f"     {emoji_calidad} Calidad del cluster: {cluster_quality_info['quality']}")
                print(f"     🎯 Score silueta cluster: {cluster_quality_info['silhouette_mean']:.4f}")
                print(f"     📉 Inercia intra-cluster: {cluster_quality_info['inertia']:.2f}")
        
        # 7. Crear visualizaciones
        print(f"\n📊 CREANDO VISUALIZACIONES:")
        
        # Gráfico detallado de silueta
        grafico_silueta_detallado(X_scaled, labels_final, k_final, nombre_archivo)
        
        # Visualización de clusters en 2D (ya no necesitamos PCA porque usamos solo 2 variables)
        print("   Saltando PCA - usando variables originales para visualización")
        
        # 8. Guardar dataset etiquetado
        print(f"\n💾 GUARDANDO DATASET ETIQUETADO:")
        
        # Crear DataFrame con etiquetas
        df_etiquetado = df.copy()
        
        # Crear columna de cluster (inicializar con -1 para filas eliminadas)
        df_etiquetado['Cluster'] = -1
        
        # Asignar etiquetas solo a las filas válidas (SOLO POSITIVOS: 0, 1, 2, ...)
        indices_validos = X_original.index
        df_etiquetado.loc[indices_validos, 'Cluster'] = labels_final
        
        # Verificar que no hay clusters negativos en el dataset final
        clusters_asignados = df_etiquetado[df_etiquetado['Cluster'] >= 0]['Cluster'].unique()
        print(f"   Clusters asignados en dataset: {sorted(clusters_asignados)}")
        
        # Crear nombre de archivo de salida
        nombre_salida = nombre_archivo.replace('07_db_', '09_db_').replace('08_db_', '09_db_')
        if not nombre_salida.startswith('09_db_'):
            nombre_salida = '09_db_' + nombre_salida
        
        ruta_salida = os.path.join(carpeta_origen, nombre_salida)
        df_etiquetado.to_csv(ruta_salida, index=False)
        
        print(f"   ✅ Dataset etiquetado guardado: {ruta_salida}")
        print(f"   📊 Dimensiones: {df_etiquetado.shape}")
        print(f"   🏷️ Filas con cluster asignado: {(df_etiquetado['Cluster'] >= 0).sum()}")
        print(f"   ❌ Filas sin cluster (datos faltantes): {(df_etiquetado['Cluster'] == -1).sum()}")
        
        # 9. CREAR BOXPLOTS DETALLADOS
        crear_boxplots_detallados(df_etiquetado, k_final, nombre_archivo, columna_objetivo)
        
        # 10. CREAR GRÁFICOS DE DISPERSIÓN POR CLUSTERS
        crear_grafico_dispersion_clusters(df_etiquetado, k_final, nombre_archivo, columna_objetivo, columna_overall)
        
        # 11. Guardar resumen de análisis
        resumen = {
            'archivo': nombre_archivo,
            'filas_totales': len(df),
            'filas_validas': len(X_original),
            'variables_clustering': f'{columna_objetivo} + {columna_overall}',
            'k_codo': k_codo,
            'k_silueta': k_silueta,
            'k_final': k_final,
            'inertia_total': inercia_final,
            'silueta_final': silueta_final
        }
        
        # Agregar estadísticas por cluster
        for i in range(k_final):
            mask = labels_final == i
            cluster_valores = X_original.iloc[mask][columna_objetivo]
            cluster_overall = X_original.iloc[mask][columna_overall]
            
            resumen[f'cluster_{i}_size'] = len(cluster_valores)
            resumen[f'cluster_{i}_valor_promedio'] = cluster_valores.mean()
            resumen[f'cluster_{i}_valor_mediana'] = cluster_valores.median()
            resumen[f'cluster_{i}_overall_promedio'] = cluster_overall.mean()
            resumen[f'cluster_{i}_overall_mediana'] = cluster_overall.median()
        
        # La información de calidad ya está incluida en calidad_info
        resumen.update(calidad_info)
        
        return resumen
        
    except Exception as e:
        print(f"❌ ERROR procesando {nombre_archivo}: {str(e)}")
        import traceback
        print(f"   Detalles del error: {traceback.format_exc()}")
        return None

def clustering_todos_archivos(carpeta="DB_separadas"):
    """
    Aplicar clustering a todos los archivos CSV en la carpeta especificada
    """
    print("🚀 INICIANDO CLUSTERING K-MEANS PARA TODOS LOS ARCHIVOS")
    print("="*60)
    print("🎯 CONFIGURACIÓN DEL ANÁLISIS:")
    print("   📊 Variables para clustering: Valor de mercado + Overall rating") 
    print("   🔍 Método de selección K: PRIORIDAD AL MÉTODO DEL CODO")
    print("   📈 Método de silueta: Solo como información complementaria")
    print("="*60)
    
    # Crear carpeta para gráficos
    crear_carpeta_kmeans()
    
    # Obtener lista de archivos CSV
    archivos_csv = [f for f in os.listdir(carpeta) if f.endswith('.csv')]
    
    if not archivos_csv:
        print(f"❌ No se encontraron archivos CSV en la carpeta {carpeta}")
        return
    
    print(f"📁 Carpeta de origen: {carpeta}")
    print(f"📊 Archivos encontrados: {len(archivos_csv)}")
    print(f"📁 Gráficos se guardarán en: k-means/")
    print(f"📁 Datasets etiquetados se guardarán en: {carpeta}/")
    
    # Procesar cada archivo
    resultados = []
    archivos_procesados = 0
    archivos_con_error = 0
    
    for archivo in archivos_csv:
        ruta_archivo = os.path.join(carpeta, archivo)
        resultado = procesar_archivo_csv(ruta_archivo, carpeta)
        
        if resultado is not None:
            resultados.append(resultado)
            archivos_procesados += 1
        else:
            archivos_con_error += 1
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📋 RESUMEN FINAL DEL CLUSTERING")
    print(f"{'='*60}")
    
    print(f"📊 Archivos procesados exitosamente: {archivos_procesados}")
    print(f"❌ Archivos con errores: {archivos_con_error}")
    print(f"📁 Total de archivos: {len(archivos_csv)}")
    
    if resultados:
        # Crear DataFrame con resumen
        df_resumen = pd.DataFrame(resultados)
        
        # Guardar resumen
        df_resumen.to_csv('k-means/00_resumen_clustering.csv', index=False)
        print(f"💾 Resumen guardado: k-means/00_resumen_clustering.csv")
        
        # Estadísticas generales
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"   K promedio seleccionado (método del codo): {df_resumen['k_final'].mean():.1f}")
        print(f"   K del codo promedio original: {df_resumen['k_codo'].mean():.1f}")
        print(f"   K de silueta promedio (referencia): {df_resumen['k_silueta'].mean():.1f}")
        print(f"   Score de silueta promedio: {df_resumen['silueta_final'].mean():.4f}")
        print(f"   Inercia promedio: {df_resumen['inertia_total'].mean():.2f}")
        print(f"   Rango de K utilizados: {df_resumen['k_final'].min()} - {df_resumen['k_final'].max()}")
        
        # Análisis de concordancia entre métodos
        concordancia = sum(df_resumen['k_codo'] == df_resumen['k_silueta'])
        total = len(df_resumen)
        print(f"   Concordancia codo-silueta: {concordancia}/{total} ({concordancia/total*100:.1f}%)")
        
        # Análisis de calidad general
        print(f"\n🏆 ANÁLISIS DE CALIDAD GENERAL:")
        calidades = df_resumen['calidad_general'].value_counts()
        for calidad, count in calidades.items():
            porcentaje = count / total * 100
            if calidad == 'EXCELENTE':
                emoji = "🌟"
            elif calidad == 'BUENA':
                emoji = "✅"
            elif calidad == 'REGULAR':
                emoji = "⚠️"
            else:
                emoji = "❌"
            print(f"   {emoji} {calidad}: {count}/{total} ({porcentaje:.1f}%)")
        
        # Estadísticas de clusters por calidad
        clusters_buenos_total = df_resumen['clusters_buenos'].sum()
        clusters_regulares_total = df_resumen['clusters_regulares'].sum()
        clusters_pobres_total = df_resumen['clusters_pobres'].sum()
        clusters_total = clusters_buenos_total + clusters_regulares_total + clusters_pobres_total
        
        print(f"\n📊 DISTRIBUCIÓN DE CALIDAD DE CLUSTERS:")
        print(f"   🟢 Clusters BUENOS: {clusters_buenos_total}/{clusters_total} ({clusters_buenos_total/clusters_total*100:.1f}%)")
        print(f"   🟡 Clusters REGULARES: {clusters_regulares_total}/{clusters_total} ({clusters_regulares_total/clusters_total*100:.1f}%)")
        print(f"   🔴 Clusters POBRES: {clusters_pobres_total}/{clusters_total} ({clusters_pobres_total/clusters_total*100:.1f}%)")
        
        # Mostrar resumen por archivo
        print(f"\n📊 RESUMEN POR ARCHIVO:")
        print("-" * 100)
        print(f"{'Archivo':<30} {'K':<3} {'Silueta':<8} {'Inercia':<8} {'Calidad':<10} {'Filas':<6} {'Clusters'}")
        print("-" * 100)
        
        for _, row in df_resumen.iterrows():
            archivo_corto = row['archivo'][:28] + '..' if len(row['archivo']) > 30 else row['archivo']
            clusters_info = f"[{', '.join([str(row[f'cluster_{i}_size']) for i in range(row['k_final'])])}]"
            
            # Emoji para calidad
            if row['calidad_general'] == 'EXCELENTE':
                emoji_cal = "🌟"
            elif row['calidad_general'] == 'BUENA':
                emoji_cal = "✅"
            elif row['calidad_general'] == 'REGULAR':
                emoji_cal = "⚠️"
            else:
                emoji_cal = "❌"
            
            calidad_mostrar = f"{emoji_cal}{row['calidad_general'][:4]}"
            
            print(f"{archivo_corto:<30} {row['k_final']:<3} {row['silueta_final']:<8.4f} {row['inertia_total']:<8.1f} {calidad_mostrar:<10} {row['filas_validas']:<6} {clusters_info}")
        
        print(f"\n🎯 PROCESO COMPLETADO:")
        print(f"   ✅ Todos los gráficos guardados en: k-means/")
        print(f"   ✅ Todos los datasets etiquetados guardados en: {carpeta}/")
        print(f"   ✅ Resumen completo en: k-means/00_resumen_clustering.csv")
        print(f"   📊 Boxplots detallados creados para cada posición")
    
    else:
        print("❌ No se pudo procesar ningún archivo correctamente")

# Ejecutar análisis principal
if __name__ == "__main__":
    # Configurar salida a archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'k-means/kmeans_{timestamp}.txt'
    
    # Crear carpeta si no existe
    os.makedirs('k-means', exist_ok=True)
    
    # Configurar redirección de salida
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print(f"📝 Salida del análisis guardándose en: {log_file}")
        print(f"🕐 Inicio del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        clustering_todos_archivos()
        
        print("="*60)
        print(f"🕐 Fin del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📝 Log completo guardado en: {log_file}")
        
    finally:
        # Restaurar salida normal
        sys.stdout = tee.terminal
        tee.close()
        print(f"\n✅ Análisis completado. Log guardado en: {log_file}") 