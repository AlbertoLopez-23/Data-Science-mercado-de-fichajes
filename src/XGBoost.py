import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings
import sys
import json
import logging
from scipy import stats
warnings.filterwarnings('ignore')

class XGBoostPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.selected_features = {}
        self.results = {}
        
        # Configurar matplotlib para gráficos
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
    
    def setup_logging(self, output_folder):
        """
        Configura el sistema de logging para capturar todos los mensajes
        """
        # Crear logger
        self.logger = logging.getLogger('XGBoostPredictor')
        self.logger.setLevel(logging.INFO)
        
        # Limpiar handlers existentes
        self.logger.handlers = []
        
        # Crear handler para archivo
        log_file_path = os.path.join(output_folder, 'logs_completos.txt')
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Crear handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Crear formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        # Formatter para consola sin timestamp
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Añadir handlers al logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_file_path = log_file_path
        return log_file_path
    
    def log_and_print(self, message):
        """
        Función para registrar mensaje tanto en consola como en archivo
        """
        if hasattr(self, 'logger'):
            self.logger.info(message)
        else:
            print(message)

    def get_features_from_dataframe(self, df, target_col='Valor de mercado actual (numérico)', cluster_col='Cluster'):
        """
        Obtiene todas las features del dataframe excepto las columnas excluidas
        """
        # Columnas a excluir
        exclude_columns = [
            'Lugar de nacimiento (país)',
            'Nacionalidad', 
            'Club actual',
            'Proveedor',
            'Fin de contrato',
            'Fecha de fichaje', 
            'comprado_por',
            target_col,  # Columna objetivo
            cluster_col  # Columna de cluster
        ]
        
        # Obtener todas las columnas disponibles
        all_columns = df.columns.tolist()
        
        # Filtrar columnas excluidas (case-insensitive)
        available_features = []
        excluded_found = []
        
        for col in all_columns:
            exclude = False
            for exc_col in exclude_columns:
                if col.lower() == exc_col.lower() or exc_col.lower() in col.lower():
                    exclude = True
                    excluded_found.append(col)
                    break
            if not exclude:
                available_features.append(col)
        
        self.log_and_print(f"Columnas excluidas encontradas: {excluded_found}")
        self.log_and_print(f"Features disponibles: {len(available_features)} de {len(all_columns)} columnas totales")
        
        return available_features

    def preprocess_features(self, df, feature_cols):
        """
        Preprocesa las features numéricas únicamente - elimina categóricas
        """
        df_processed = df[feature_cols].copy()
        
        # Información sobre el procesamiento
        categorical_cols = []
        numerical_cols = []
        
        for col in feature_cols:
            if df_processed[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        self.log_and_print(f"Columnas categóricas encontradas (serán eliminadas): {len(categorical_cols)}")
        self.log_and_print(f"Columnas numéricas: {len(numerical_cols)}")
        
        # Eliminar columnas categóricas completamente
        if categorical_cols:
            self.log_and_print(f"Eliminando columnas categóricas: {categorical_cols}")
            df_processed = df_processed.drop(columns=categorical_cols)
            numerical_cols = [col for col in feature_cols if col not in categorical_cols]
        
        # Manejar valores nulos en columnas numéricas
        for col in numerical_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        return df_processed, {}  # No hay label encoders

    def load_and_preprocess_data(self, file_path):
        """
        Carga y preprocesa los datos de un archivo específico
        """
        try:
            # Cargar datos
            df = pd.read_csv(file_path)
            self.log_and_print(f"Archivo cargado: {file_path}")
            self.log_and_print(f"Shape original: {df.shape}")
            
            # Verificar que existe la variable objetivo
            target_column = 'Valor de mercado actual (numérico)'
            if target_column not in df.columns:
                self.log_and_print(f"Variable objetivo '{target_column}' no encontrada")
                return None, None, None, None
            
            # Verificar que existe la columna de cluster
            cluster_column = 'Cluster'
            if cluster_column not in df.columns:
                self.log_and_print(f"Columna de cluster '{cluster_column}' no encontrada")
                return None, None, None, None
            
            # Obtener todas las features disponibles excepto las excluidas
            available_features = self.get_features_from_dataframe(df, target_column, cluster_column)
            
            if not available_features:
                self.log_and_print("No se encontraron features válidas para el modelo")
                return None, None, None, None
            
            self.log_and_print(f"Features seleccionadas: {len(available_features)}")
            for i, feature in enumerate(available_features[:10], 1):  # Mostrar solo las primeras 10
                self.log_and_print(f"  {i}. {feature}")
            if len(available_features) > 10:
                self.log_and_print(f"  ... y {len(available_features) - 10} más")
            
            return df, available_features, target_column, cluster_column
            
        except Exception as e:
            self.log_and_print(f"Error cargando archivo: {str(e)}")
            return None, None, None, None

    def create_predictions_csv(self, df, available_features, target_column, cluster_column, file_path):
        """
        Crea un CSV con todas las predicciones para el dataset completo
        """
        self.log_and_print("Generando predicciones para todo el dataset...")
        
        # Crear copia del dataframe original
        df_with_predictions = df.copy()
        df_with_predictions['Valor_Predicho'] = np.nan
        
        # Procesar cada cluster
        for cluster_id in sorted(df[cluster_column].unique()):
            self.log_and_print(f"Generando predicciones para Cluster {cluster_id}...")
            
            try:
                # Filtrar datos del cluster
                cluster_mask = df[cluster_column] == cluster_id
                cluster_data = df[cluster_mask].copy()
                
                # Preprocesar features
                processed_features, _ = self.preprocess_features(cluster_data, available_features)
                
                # Verificar si tenemos el modelo entrenado para este cluster
                model_key = f'cluster_{cluster_id}'
                if model_key not in self.models:
                    self.log_and_print(f"⚠️  Modelo no encontrado para Cluster {cluster_id}")
                    continue
                
                model = self.models[model_key]
                scaler = self.scalers[model_key]
                
                # Usar las mismas features que se usaron para entrenar
                X = processed_features.copy()
                y = cluster_data[target_column].copy()
                
                # Las variables X NO se escalan (según requerimiento)
                X_for_prediction = X.copy()
                
                # Escalar la variable objetivo para predicción
                y_scaled = scaler.transform(y.values.reshape(-1, 1)).flatten()
                
                # Generar predicciones (el modelo predice en escala escalada)
                y_pred_scaled = model.predict(X_for_prediction)
                
                # Convertir de vuelta a escala original
                y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                # Asignar predicciones al dataframe
                df_with_predictions.loc[cluster_mask, 'Valor_Predicho'] = y_pred
                
                self.log_and_print(f"✓ Predicciones generadas para Cluster {cluster_id}: {len(y_pred)} valores")
                
            except Exception as e:
                self.log_and_print(f"✗ Error generando predicciones para Cluster {cluster_id}: {str(e)}")
        
        # Guardar CSV con predicciones
        base_name = os.path.basename(file_path).replace('.csv', '')
        predictions_csv_path = os.path.join(self.output_folder, f'{base_name}_con_predicciones.csv')
        df_with_predictions.to_csv(predictions_csv_path, index=False)
        
        # Estadísticas de predicciones
        total_predictions = df_with_predictions['Valor_Predicho'].notna().sum()
        total_samples = len(df_with_predictions)
        
        self.log_and_print(f"CSV con predicciones guardado: {predictions_csv_path}")
        self.log_and_print(f"Predicciones generadas: {total_predictions}/{total_samples} ({total_predictions/total_samples*100:.1f}%)")
        
        return predictions_csv_path, df_with_predictions

    def create_comprehensive_plots(self, y_true, y_pred, cluster_id, split_name, model_info):
        """
        Crea un conjunto completo de gráficas para análisis del modelo (igual que SVR.py)
        """
        # Configurar subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Scatter plot de predicciones vs reales
        ax1 = plt.subplot(3, 3, 1)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        plt.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('Valores Reales')
        plt.ylabel('Valores Predichos')
        plt.title(f'Reales vs Predichos - Cluster {cluster_id} ({split_name})')
        plt.grid(True, alpha=0.3)
        
        # Calcular R²
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 2. Residuales vs predichos
        ax2 = plt.subplot(3, 3, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Valores Predichos')
        plt.ylabel('Residuales')
        plt.title(f'Residuales vs Predichos - Cluster {cluster_id}')
        plt.grid(True, alpha=0.3)
        
        # 3. Histograma de residuales
        ax3 = plt.subplot(3, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Residuales')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de Residuales - Cluster {cluster_id}')
        plt.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax4 = plt.subplot(3, 3, 4)
        stats.probplot(residuals, dist="norm", plot=ax4)
        plt.title(f'Q-Q Plot - Cluster {cluster_id}')
        plt.grid(True, alpha=0.3)
        
        # 5. Boxplot de residuales
        ax5 = plt.subplot(3, 3, 5)
        plt.boxplot(residuals, vert=True)
        plt.ylabel('Residuales')
        plt.title(f'Boxplot Residuales - Cluster {cluster_id}')
        plt.grid(True, alpha=0.3)
        
        # 6. Serie temporal de residuales (orden de predicción)
        ax6 = plt.subplot(3, 3, 6)
        plt.plot(range(len(residuals)), residuals, alpha=0.7, color='purple')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Orden de observación')
        plt.ylabel('Residuales')
        plt.title(f'Serie Temporal Residuales - Cluster {cluster_id}')
        plt.grid(True, alpha=0.3)
        
        # 7. Distribución de valores reales vs predichos
        ax7 = plt.subplot(3, 3, 7)
        plt.hist(y_true, bins=20, alpha=0.5, label='Reales', color='blue')
        plt.hist(y_pred, bins=20, alpha=0.5, label='Predichos', color='red')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución Valores - Cluster {cluster_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Métricas textuales
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        # Calcular métricas
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calcular MAPE (Mean Absolute Percentage Error) de forma robusta
        def safe_mape(y_true, y_pred):
            """Calcula MAPE evitando divisiones por cero"""
            # Filtrar valores donde y_true es muy pequeño (cerca de cero)
            mask = np.abs(y_true) > np.abs(y_true).mean() * 0.01  # Al menos 1% de la media
            if mask.sum() == 0:  # Si todos los valores son muy pequeños
                return np.nan
            
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            
            return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        
        mape = safe_mape(y_true, y_pred)
        
        metrics_text = f"""
        MÉTRICAS DEL MODELO - CLUSTER {cluster_id}
        {split_name.upper()}
        
        R² Score: {r2:.4f}
        MAE: {mae:,.2f}
        MSE: {mse:,.2f}
        RMSE: {rmse:,.2f}
        MAPE: {mape:.2f}% {'' if not np.isnan(mape) else '(N/A)'}
        
        Muestras: {len(y_true)}
        Media Real: {y_true.mean():,.2f}
        Media Pred: {y_pred.mean():,.2f}
        
        Parámetros del modelo:
        n_estimators: {model_info.get('n_estimators', 'N/A')}
        max_depth: {model_info.get('max_depth', 'N/A')}
        min_child_weight: {model_info.get('min_child_weight', 'N/A')}
        gamma: {model_info.get('gamma', 'N/A')}
        subsample: {model_info.get('subsample', 'N/A')}
        colsample_bytree: {model_info.get('colsample_bytree', 'N/A')}
        learning_rate: {model_info.get('learning_rate', 'N/A')}
        """
        
        ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        
        # 9. Análisis de errores por percentiles
        ax9 = plt.subplot(3, 3, 9)
        abs_errors = np.abs(residuals)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        error_percentiles = [np.percentile(abs_errors, p) for p in percentiles]
        
        plt.bar(range(len(percentiles)), error_percentiles, color='coral', alpha=0.7)
        plt.xticks(range(len(percentiles)), [f'P{p}' for p in percentiles])
        plt.xlabel('Percentil')
        plt.ylabel('Error Absoluto')
        plt.title(f'Errores por Percentil - Cluster {cluster_id}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_filename = os.path.join(self.output_folder, 'graficas', 
                                   f'analisis_completo_cluster_{cluster_id}_{split_name.lower()}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_filename, {
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mean_real': y_true.mean(),
            'mean_pred': y_pred.mean(),
            'residuals_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'min': residuals.min(),
                'max': residuals.max()
            }
        }

    def create_feature_importance_plot(self, model, feature_names, cluster_id):
        """
        Crea gráfico de importancia de features para XGBoost
        """
        # Importancia de features de XGBoost
        importances = model.feature_importances_
        
        # Verificar que las dimensiones coincidan
        if len(importances) != len(feature_names):
            self.log_and_print(f"⚠️  Advertencia: Dimensiones no coinciden - importances: {len(importances)}, features: {len(feature_names)}")
            # Tomar el mínimo para evitar errores
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]
        
        # Crear DataFrame para facilitar el ordenamiento
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Tomar las top 20 features
        top_features = feature_importance_df.tail(20)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia (Feature Importance)')
        plt.title(f'Top 20 Features Más Importantes - Cluster {cluster_id}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_filename = os.path.join(self.output_folder, 'graficas', 
                                   f'feature_importance_cluster_{cluster_id}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_filename, top_features.to_dict('records')

    def create_comparison_plots(self, summary_df, all_results):
        """
        Crea gráficos comparativos entre clusters
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Comparación Train vs Test R²
        x = np.arange(len(summary_df))
        width = 0.35
        axes[0, 0].bar(x - width/2, summary_df['Train_R2'], width, label='Train R²', color='lightblue')
        axes[0, 0].bar(x + width/2, summary_df['Test_R2'], width, label='Test R²', color='darkblue')
        axes[0, 0].set_title('Comparación Train vs Test R²')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(summary_df['Cluster'].astype(str))
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Comparación Train vs Test RMSE
        axes[0, 1].bar(x - width/2, summary_df['Train_RMSE'], width, label='Train RMSE', color='lightcoral')
        axes[0, 1].bar(x + width/2, summary_df['Test_RMSE'], width, label='Test RMSE', color='darkred')
        axes[0, 1].set_title('Comparación Train vs Test RMSE')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(summary_df['Cluster'].astype(str))
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Número de muestras por cluster
        axes[0, 2].bar(summary_df['Cluster'].astype(str), summary_df['N_Muestras'], color='teal')
        axes[0, 2].set_title('Número de Muestras por Cluster')
        axes[0, 2].set_xlabel('Cluster')
        axes[0, 2].set_ylabel('Número de Muestras')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Ratio de Overfitting por cluster
        axes[1, 0].bar(summary_df['Cluster'].astype(str), summary_df['Overfitting_Ratio'], color='coral')
        axes[1, 0].set_title('Ratio de Overfitting por Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Ratio Train/Test')
        axes[1, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Umbral Alto')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Comparación Train vs Test MAE
        axes[1, 1].bar(x - width/2, summary_df['Train_MAE'], width, label='Train MAE', color='lightgreen')
        axes[1, 1].bar(x + width/2, summary_df['Test_MAE'], width, label='Test MAE', color='darkgreen')
        axes[1, 1].set_title('Comparación Train vs Test MAE')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(summary_df['Cluster'].astype(str))
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Dispersión de valores reales vs predichos (Test)
        axes[1, 2].set_title('Valores Reales vs Predichos (Test)')
        axes[1, 2].set_xlabel('Valores Reales')
        axes[1, 2].set_ylabel('Valores Predichos')
        
        # Recopilar todos los valores reales y predichos de test
        all_y_true = []
        all_y_pred = []
        cluster_labels = []
        
        for cluster_id, result in all_results.items():
            # Los datos de test están en las métricas, pero necesitamos acceso a los valores originales
            # Por ahora, creamos datos simulados basados en las métricas
            if 'test_data' in result:
                y_true = result['test_data']['y_true']
                y_pred = result['test_data']['y_pred']
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                cluster_labels.extend([cluster_id] * len(y_true))
        
        if all_y_true and all_y_pred:
            # Crear scatter plot coloreado por cluster
            unique_clusters = list(set(cluster_labels))
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster in enumerate(unique_clusters):
                mask = [c == cluster for c in cluster_labels]
                y_true_cluster = [y for j, y in enumerate(all_y_true) if mask[j]]
                y_pred_cluster = [y for j, y in enumerate(all_y_pred) if mask[j]]
                
                axes[1, 2].scatter(y_true_cluster, y_pred_cluster, 
                                 alpha=0.6, label=f'Cluster {cluster}', 
                                 color=colors[i], s=30)
            
            # Línea de referencia perfecta
            min_val = min(min(all_y_true), min(all_y_pred))
            max_val = max(max(all_y_true), max(all_y_pred))
            axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Si no hay datos disponibles, mostrar mensaje
            axes[1, 2].text(0.5, 0.5, 'Datos de predicciones\nno disponibles', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 2].transAxes, fontsize=12)
        
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(self.output_folder, 'graficas', 'comparacion_clusters.png')
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_and_print(f"Gráfico comparativo guardado en: {comparison_plot_path}")
        return comparison_plot_path

    def train_xgboost_model_for_cluster(self, X, y, cluster_id, feature_names):
        """
        Entrena un modelo XGBoost para un cluster específico
        """
        self.log_and_print(f"\nEntrenando modelo XGBoost para Cluster {cluster_id}...")
        self.log_and_print(f"Muestras: {len(y)}, Features: {X.shape[1]}")
        
        # Análisis preliminar del cluster
        y_std = y.std()
        y_mean = y.mean()
        cv_y = y_std / y_mean if y_mean > 0 else float('inf')
        
        self.log_and_print(f"Estadísticas del cluster:")
        self.log_and_print(f"  Media: {y_mean:,.0f}")
        self.log_and_print(f"  Desviación estándar: {y_std:,.0f}")
        self.log_and_print(f"  Coeficiente de variación: {cv_y:.3f}")
        
        # Verificar que hay suficientes datos para entrenar
        if len(X) < 10:
            self.log_and_print(f"Warning: Solo {len(X)} muestras disponibles. Muy pocas para entrenar un modelo robusto.")
            return None
        
        # División train/test
        test_size = min(0.25, max(0.1, 1/len(X) * 5))  # Ajustar test_size para datasets pequeños
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Aplicar StandardScaler SOLO a la variable objetivo (y)
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Las variables X NO se escalan (según requerimiento)
        X_train_final = X_train.copy()
        X_test_final = X_test.copy()
        
        self.log_and_print(f"Variables X SIN escalar - Shape: {X_train_final.shape}")
        self.log_and_print(f"Variable y escalada - Rango train: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
        
        self.log_and_print(f"Configuración:")
        self.log_and_print(f"  División train/test: {len(X_train)}/{len(X_test)} ({(1-test_size)*100:.0f}%/{test_size*100:.0f}%)")
        
        # Crear y entrenar el modelo XGBoost
        xgb_params = {
            'n_estimators': 50,
            'max_depth': 4,  # Reducido drásticamente de 6 a 4
            'min_child_weight': 5,  # Aumentado de 1 (default) a 5
            'gamma': 0.2,  # Añadido para controlar ganancia mínima
            'subsample': 0.8,  # Añadido para usar solo 80% de los datos
            'colsample_bytree': 0.8,  # Añadido para usar solo 80% de las features
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.log_and_print("Entrenando modelo XGBoost...")
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train_final, y_train_scaled)
        
        # Predicciones en escala escalada
        self.log_and_print("Generando predicciones...")
        y_train_pred_scaled = model.predict(X_train_final)
        y_test_pred_scaled = model.predict(X_test_final)
        
        # Convertir de vuelta a escala original
        y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calcular métricas finales
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calcular métricas adicionales
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        # Calcular MAPE (Mean Absolute Percentage Error) de forma robusta
        def safe_mape(y_true, y_pred):
            """Calcula MAPE evitando divisiones por cero"""
            # Filtrar valores donde y_true es muy pequeño (cerca de cero)
            mask = np.abs(y_true) > np.abs(y_true).mean() * 0.01  # Al menos 1% de la media
            if mask.sum() == 0:  # Si todos los valores son muy pequeños
                return np.nan
            
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            
            return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        
        train_mape = safe_mape(y_train, y_train_pred)
        test_mape = safe_mape(y_test, y_test_pred)
        
        # Calcular ratio de overfitting
        if test_r2 > 0:
            overfitting_ratio = train_r2 / test_r2
        elif test_r2 == 0:
            overfitting_ratio = float('inf') if train_r2 > 0 else 1.0
        else:  # test_r2 < 0
            overfitting_ratio = abs(train_r2 / test_r2) if test_r2 != 0 else float('inf')
        
        self.log_and_print(f"Métricas finales:")
        self.log_and_print(f"  R² Train: {train_r2:.4f}")
        self.log_and_print(f"  R² Test: {test_r2:.4f}")
        self.log_and_print(f"  MAE Train: {train_mae:,.2f}")
        self.log_and_print(f"  MAE Test: {test_mae:,.2f}")
        self.log_and_print(f"  MSE Train: {train_mse:,.2f}")
        self.log_and_print(f"  MSE Test: {test_mse:,.2f}")
        self.log_and_print(f"  RMSE Train: {train_rmse:,.2f}")
        self.log_and_print(f"  RMSE Test: {test_rmse:,.2f}")
        self.log_and_print(f"  MAPE Train: {train_mape:.2f}%" if not np.isnan(train_mape) else "  MAPE Train: N/A")
        self.log_and_print(f"  MAPE Test: {test_mape:.2f}%" if not np.isnan(test_mape) else "  MAPE Test: N/A")
        self.log_and_print(f"  Ratio overfitting: {overfitting_ratio:.2f}")
        
        # Generar gráficas comprehensivas
        self.log_and_print("Generando gráficas...")
        train_plot, train_metrics = self.create_comprehensive_plots(
            y_train, y_train_pred, cluster_id, "Train", xgb_params
        )
        test_plot, test_metrics = self.create_comprehensive_plots(
            y_test, y_test_pred, cluster_id, "Test", xgb_params
        )
        
        # Generar gráfico de importancia de features
        importance_plot, importance_data = self.create_feature_importance_plot(
            model, feature_names, cluster_id
        )
        
        # Guardar modelo y scaler
        model_key = f'cluster_{cluster_id}'
        self.models[model_key] = model
        self.scalers[model_key] = y_scaler
        self.selected_features[model_key] = feature_names
        
        # Métricas finales
        metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'overfitting_ratio': overfitting_ratio,
            'n_samples': len(y),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'features_used': len(feature_names),
            'model_params': xgb_params
        }
        
        result = {
            'model': model,
            'scaler': y_scaler,
            'metrics': metrics,
            'train_plot': train_plot,
            'test_plot': test_plot,
            'importance_plot': importance_plot,
            'importance_data': importance_data,
            'feature_names': feature_names,
            'test_data': {
                'y_true': y_test.values,
                'y_pred': y_test_pred
            }
        }
        
        self.log_and_print(f"✓ Modelo entrenado exitosamente para Cluster {cluster_id}")
        
        return result

    def process_single_file(self, file_path):
        """
        Procesa un archivo CSV específico
        """
        self.log_and_print(f"\n{'='*80}")
        self.log_and_print(f"PROCESANDO ARCHIVO: {os.path.basename(file_path)}")
        self.log_and_print(f"{'='*80}")
        
        # Cargar y preprocesar datos
        df, available_features, target_column, cluster_column = self.load_and_preprocess_data(file_path)
        
        if df is None:
            self.log_and_print("Error: No se pudieron cargar los datos")
            return None
        
        # Analizar distribución de clusters
        cluster_counts = df[cluster_column].value_counts().sort_index()
        self.log_and_print(f"\nDistribución de clusters:")
        for cluster_id, count in cluster_counts.items():
            self.log_and_print(f"  Cluster {cluster_id}: {count} muestras")
        
        # Resultados por cluster
        results = {}
        
        # Procesar cada cluster
        for cluster_id in sorted(df[cluster_column].unique()):
            self.log_and_print(f"\n{'-'*60}")
            self.log_and_print(f"PROCESANDO CLUSTER {cluster_id}")
            self.log_and_print(f"{'-'*60}")
            
            # Filtrar datos del cluster
            cluster_data = df[df[cluster_column] == cluster_id].copy()
            
            # Preprocesar features
            processed_features, _ = self.preprocess_features(cluster_data, available_features)
            
            if processed_features.empty:
                self.log_and_print(f"⚠️  No hay features válidas para el Cluster {cluster_id}")
                continue
            
            # Preparar datos para entrenamiento
            X = processed_features
            y = cluster_data[target_column]
            
            # Entrenar modelo
            result = self.train_xgboost_model_for_cluster(X, y, cluster_id, processed_features.columns.tolist())
            
            if result:
                results[cluster_id] = result
                self.log_and_print(f"✓ Cluster {cluster_id} procesado exitosamente")
            else:
                self.log_and_print(f"✗ Error procesando Cluster {cluster_id}")
        
        if not results:
            self.log_and_print("⚠️  No se pudo procesar ningún cluster")
            return None
        
        # Generar CSV con predicciones
        self.log_and_print(f"\n{'-'*60}")
        self.log_and_print("GENERANDO ARCHIVO CON PREDICCIONES")
        self.log_and_print(f"{'-'*60}")
        
        predictions_csv, df_with_predictions = self.create_predictions_csv(
            df, available_features, target_column, cluster_column, file_path
        )
        
        return {
            'results': results,
            'cluster_counts': cluster_counts,
            'predictions_csv': predictions_csv,
            'df_with_predictions': df_with_predictions,
            'total_features': len(available_features),
            'total_samples': len(df)
        }

    def setup_output_folder(self, file_path):
        """
        Configura la carpeta de salida para los resultados
        """
        base_name = os.path.basename(file_path).replace('.csv', '')
        self.output_folder = f'XGBoost_{base_name}'
        
        # Crear carpetas
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, 'graficas'), exist_ok=True)
        
        return self.output_folder

    def generate_final_report(self, results, file_path, cluster_counts):
        """
        Genera reporte final con resumen de resultados
        """
        self.log_and_print(f"\n{'='*80}")
        self.log_and_print("REPORTE FINAL")
        self.log_and_print(f"{'='*80}")
        
        base_name = os.path.basename(file_path).replace('.csv', '')
        
        self.log_and_print(f"Dataset: {base_name}")
        self.log_and_print(f"Total de muestras: {results['total_samples']}")
        self.log_and_print(f"Features utilizadas: {results['total_features']}")
        self.log_and_print(f"Clusters procesados: {len(results['results'])}")
        
        # Resumen por cluster
        self.log_and_print(f"\nRESUMEN POR CLUSTER:")
        self.log_and_print(f"{'Cluster':<8} {'Muestras':<10} {'R² Test':<10} {'RMSE Test':<12} {'MAE Test':<12}")
        self.log_and_print(f"{'-'*60}")
        
        test_r2_scores = []
        test_rmse_scores = []
        test_mae_scores = []
        
        # Crear DataFrame para gráficos comparativos y reporte detallado
        summary_data = []
        
        for cluster_id, result in results['results'].items():
            metrics = result['metrics']
            test_r2 = metrics['test_r2']
            test_rmse = metrics['test_rmse']
            test_mae = metrics['test_mae']
            train_r2 = metrics['train_r2']
            train_rmse = metrics['train_rmse']
            train_mae = metrics['train_mae']
            overfitting_ratio = metrics['overfitting_ratio']
            
            test_r2_scores.append(test_r2)
            test_rmse_scores.append(test_rmse)
            test_mae_scores.append(test_mae)
            
            # Añadir datos para el DataFrame de resumen
            summary_data.append({
                'Cluster': cluster_id,
                'N_Muestras': metrics['n_samples'],
                'N_Features': metrics['features_used'],
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_MSE': metrics['train_mse'],
                'Test_MSE': metrics['test_mse'],
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAPE': metrics['train_mape'],
                'Test_MAPE': metrics['test_mape'],
                'Overfitting_Ratio': overfitting_ratio,
                'N_Estimators': metrics['model_params']['n_estimators'],
                'Max_Depth': metrics['model_params']['max_depth'],
                'Min_Child_Weight': metrics['model_params']['min_child_weight'],
                'Gamma': metrics['model_params']['gamma'],
                'Subsample': metrics['model_params']['subsample'],
                'Colsample_Bytree': metrics['model_params']['colsample_bytree'],
                'Learning_Rate': metrics['model_params']['learning_rate']
            })
            
            self.log_and_print(f"{cluster_id:<8} {metrics['n_samples']:<10} {test_r2:<10.4f} {test_rmse:<12.0f} {test_mae:<12.0f}")
        
        # Crear DataFrame de resumen y ordenar por Test_R2
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_R2', ascending=False)
        
        # Guardar resumen CSV
        summary_csv_path = os.path.join(self.output_folder, 'resumen_modelos_por_cluster.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        self.log_and_print(f"\nResumen CSV guardado en: {summary_csv_path}")
        
        # Estadísticas generales
        if test_r2_scores:
            self.log_and_print(f"\nESTADÍSTICAS GENERALES:")
            self.log_and_print(f"R² Test promedio: {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
            self.log_and_print(f"RMSE Test promedio: {np.mean(test_rmse_scores):,.0f} ± {np.std(test_rmse_scores):,.0f}")
            self.log_and_print(f"MAE Test promedio: {np.mean(test_mae_scores):,.0f} ± {np.std(test_mae_scores):,.0f}")
            
            best_cluster = max(results['results'].items(), key=lambda x: x[1]['metrics']['test_r2'])
            worst_cluster = min(results['results'].items(), key=lambda x: x[1]['metrics']['test_r2'])
            
            self.log_and_print(f"Mejor cluster (R²): {best_cluster[0]} (R² = {best_cluster[1]['metrics']['test_r2']:.4f})")
            self.log_and_print(f"Peor cluster (R²): {worst_cluster[0]} (R² = {worst_cluster[1]['metrics']['test_r2']:.4f})")
        
        # Generar gráfico de comparación entre clusters
        self.log_and_print(f"\nGenerando gráfico de comparación entre clusters...")
        try:
            comparison_plot_path = self.create_comparison_plots(summary_df, results['results'])
            self.log_and_print(f"✓ Gráfico de comparación generado exitosamente")
        except Exception as e:
            self.log_and_print(f"⚠️  Error generando gráfico de comparación: {str(e)}")
        
        # Generar reporte detallado en archivo de texto
        report_path = os.path.join(self.output_folder, 'reporte_detallado.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DETALLADO - MODELOS XGBOOST POR CLUSTER\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Archivo procesado: {file_path}\n")
            f.write(f"Fecha de procesamiento: {pd.Timestamp.now()}\n\n")
            
            f.write(f"CONFIGURACIÓN DEL MODELO:\n")
            f.write(f"Algoritmo: XGBoost Regressor\n")
            if summary_data:
                sample_params = metrics['model_params']
                f.write(f"Parámetros utilizados:\n")
                f.write(f"  - n_estimators: {sample_params['n_estimators']}\n")
                f.write(f"  - max_depth: {sample_params['max_depth']}\n")
                f.write(f"  - min_child_weight: {sample_params['min_child_weight']}\n")
                f.write(f"  - gamma: {sample_params['gamma']}\n")
                f.write(f"  - subsample: {sample_params['subsample']}\n")
                f.write(f"  - colsample_bytree: {sample_params['colsample_bytree']}\n")
                f.write(f"  - learning_rate: {sample_params['learning_rate']}\n")
            f.write("\n")
            
            f.write(f"DISTRIBUCIÓN DE CLUSTERS:\n")
            for cluster_id, count in cluster_counts.items():
                f.write(f"  Cluster {cluster_id}: {count} muestras\n")
            f.write("\n")
            
            f.write(f"RESUMEN DE RESULTADOS:\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            f.write(f"ESTADÍSTICAS GENERALES:\n")
            f.write(f"Mejor R² Test: {summary_df['Test_R2'].max():.4f} (Cluster {summary_df.loc[summary_df['Test_R2'].idxmax(), 'Cluster']})\n")
            f.write(f"Peor R² Test: {summary_df['Test_R2'].min():.4f} (Cluster {summary_df.loc[summary_df['Test_R2'].idxmin(), 'Cluster']})\n")
            f.write(f"R² Test promedio: {summary_df['Test_R2'].mean():.4f}\n")
            f.write(f"Desviación estándar R² Test: {summary_df['Test_R2'].std():.4f}\n")
            f.write(f"RMSE Test promedio: {summary_df['Test_RMSE'].mean():,.2f}\n")
            f.write(f"MAE Test promedio: {summary_df['Test_MAE'].mean():,.2f}\n")
            f.write(f"Clusters procesados exitosamente: {len(summary_df)}\n")
            f.write(f"Total de muestras procesadas: {summary_df['N_Muestras'].sum()}\n")
            f.write(f"Total de features utilizadas: {results['total_features']}\n\n")
            
            f.write(f"ANÁLISIS DE OVERFITTING:\n")
            avg_overfitting = summary_df['Overfitting_Ratio'].mean()
            high_overfitting = summary_df[summary_df['Overfitting_Ratio'] > 2.0]
            f.write(f"Ratio de overfitting promedio: {avg_overfitting:.2f}\n")
            f.write(f"Clusters con alto overfitting (>2.0): {len(high_overfitting)}\n")
            if len(high_overfitting) > 0:
                f.write(f"Clusters problemáticos: {', '.join(map(str, high_overfitting['Cluster'].tolist()))}\n")
            f.write("\n")
            
            f.write(f"DETALLES POR CLUSTER:\n")
            f.write(f"-" * 50 + "\n")
            for _, row in summary_df.iterrows():
                f.write(f"CLUSTER {row['Cluster']}:\n")
                f.write(f"  Muestras: {row['N_Muestras']}\n")
                f.write(f"  Features: {row['N_Features']}\n")
                f.write(f"  R² Train: {row['Train_R2']:.4f}\n")
                f.write(f"  R² Test: {row['Test_R2']:.4f}\n")
                f.write(f"  RMSE Train: {row['Train_RMSE']:,.2f}\n")
                f.write(f"  RMSE Test: {row['Test_RMSE']:,.2f}\n")
                f.write(f"  MAE Train: {row['Train_MAE']:,.2f}\n")
                f.write(f"  MAE Test: {row['Test_MAE']:,.2f}\n")
                if not pd.isna(row['Train_MAPE']):
                    f.write(f"  MAPE Train: {row['Train_MAPE']:.2f}%\n")
                if not pd.isna(row['Test_MAPE']):
                    f.write(f"  MAPE Test: {row['Test_MAPE']:.2f}%\n")
                f.write(f"  Ratio Overfitting: {row['Overfitting_Ratio']:.2f}\n")
                f.write(f"  Estado: {'⚠️ Alto overfitting' if row['Overfitting_Ratio'] > 2.0 else '✓ Overfitting controlado'}\n")
                f.write("\n")
        
        self.log_and_print(f"Reporte detallado guardado en: {report_path}")
        
        # Información de archivos generados
        self.log_and_print(f"\nARCHIVOS GENERADOS:")
        self.log_and_print(f"- Carpeta principal: {self.output_folder}/")
        self.log_and_print(f"- Logs completos: {self.output_folder}/logs_completos.txt")
        self.log_and_print(f"- Reporte detallado: {report_path}")
        self.log_and_print(f"- Resumen CSV: {summary_csv_path}")
        self.log_and_print(f"- Datos con predicciones: {results['predictions_csv']}")
        self.log_and_print(f"- Gráficas: {self.output_folder}/graficas/")
        
        # Contar archivos de gráficas generados
        graficas_dir = os.path.join(self.output_folder, 'graficas')
        if os.path.exists(graficas_dir):
            num_graficas = len([f for f in os.listdir(graficas_dir) if f.endswith('.png')])
            self.log_and_print(f"  * {num_graficas} gráficas generadas")
        
        self.log_and_print(f"\n{'='*80}")
        self.log_and_print("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        self.log_and_print(f"{'='*80}")
        
        return summary_df

def main():
    """
    Función principal que procesa un dataset específico
    """
    # Verificar argumentos de línea de comandos
    if len(sys.argv) != 2:
        print("Uso: python XGBoost.py <nombre_de_archivo>")
        print("Ejemplo: python XGBoost.py dataset.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    base_name = filename.replace('.csv', '') if filename.endswith('.csv') else filename
    
    # Directorio de datos
    db_dir = 'DB_separadas'
    
    # Construir ruta del archivo
    csv_file = os.path.join(db_dir, f'{base_name}.csv')
    
    # Verificar que el archivo existe
    if not os.path.exists(csv_file):
        print(f"Error: No se encuentra el archivo {csv_file}")
        sys.exit(1)
    
    # Crear instancia del predictor
    predictor = XGBoostPredictor()
    
    # Configurar carpeta de salida
    output_folder = predictor.setup_output_folder(csv_file)
    
    # Configurar logging
    log_file = predictor.setup_logging(output_folder)
    
    predictor.log_and_print(f"Iniciando procesamiento de XGBoost")
    predictor.log_and_print(f"Archivo: {csv_file}")
    predictor.log_and_print(f"Carpeta de salida: {output_folder}")
    predictor.log_and_print(f"Log file: {log_file}")
    
    try:
        # Procesar archivo
        results = predictor.process_single_file(csv_file)
        
        if results:
            # Generar reporte final
            summary_df = predictor.generate_final_report(results, csv_file, results['cluster_counts'])
        else:
            predictor.log_and_print("Error: No se pudieron procesar los datos")
            sys.exit(1)
            
    except Exception as e:
        predictor.log_and_print(f"Error inesperado: {str(e)}")
        import traceback
        predictor.log_and_print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()