import pandas as pd
import numpy as np
import os
import sys
import glob
import logging
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SVRPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.selected_features = {}
        self.results = {}
        
        # Configurar matplotlib para gráficos
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
    
    def setup_logging(self, output_folder):
        """
        Configura el sistema de logging para capturar todos los mensajes
        """
        # Crear logger
        self.logger = logging.getLogger('SVRPredictor')
        self.logger.setLevel(logging.INFO)
        
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
        Preprocesa las features numéricas únicamente
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
                
                # Escalar la variable objetivo para predicción
                y_scaled = scaler.transform(y.values.reshape(-1, 1)).flatten()
                
                # Generar predicciones
                y_pred_scaled = model.predict(X)
                
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
        Crea un conjunto completo de gráficas para análisis del modelo
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
        
        train_mape = safe_mape(y_true, y_pred)
        test_mape = safe_mape(y_true, y_pred)
        
        metrics_text = f"""
        MÉTRICAS DEL MODELO - CLUSTER {cluster_id}
        {split_name.upper()}
        
        R² Score: {r2:.4f}
        MAE: {mae:,.2f}
        MSE: {mse:,.2f}
        RMSE: {rmse:,.2f}
        MAPE: {train_mape:.2f}% {'' if not np.isnan(train_mape) else '(N/A)'}
        
        Muestras: {len(y_true)}
        Media Real: {y_true.mean():,.2f}
        Media Pred: {y_pred.mean():,.2f}
        
        Parámetros del modelo:
        C: {model_info.get('C', 'N/A')}
        Kernel: {model_info.get('kernel', 'N/A')}
        Gamma: {model_info.get('gamma', 'N/A')}
        Epsilon: {model_info.get('epsilon', 'N/A')}
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
            'mape': train_mape,
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
        Crea gráfico de importancia de features para modelos lineales
        """
        if hasattr(model, 'coef_'):
            # Manejar coeficientes multidimensionales
            coef = model.coef_
            if coef.ndim > 1:
                # Si es multidimensional, tomar la primera fila o aplanar
                importances = np.abs(coef.flatten())
            else:
                importances = np.abs(coef)
            
            # Verificar que las dimensiones coincidan
            if len(importances) != len(feature_names):
                self.log_and_print(f"⚠️  Advertencia: Dimensiones no coinciden - coef: {len(importances)}, features: {len(feature_names)}")
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
            plt.xlabel('Importancia (Valor Absoluto del Coeficiente)')
            plt.title(f'Top 20 Features Más Importantes - Cluster {cluster_id}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_filename = os.path.join(self.output_folder, 'graficas', 
                                       f'feature_importance_cluster_{cluster_id}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_filename, top_features.to_dict('records')
        
        return None, None
    
    def train_svr_model_for_cluster(self, X, y, cluster_id, feature_names):
        """
        Entrena un modelo SVR para un cluster específico
        """
        self.log_and_print(f"\nEntrenando modelo para Cluster {cluster_id}...")
        self.log_and_print(f"Muestras: {len(y)}, Features: {X.shape[1]}")
        
        # Análisis preliminar del cluster
        y_std = y.std()
        y_mean = y.mean()
        cv_y = y_std / y_mean if y_mean > 0 else float('inf')
        
        self.log_and_print(f"Estadísticas del cluster:")
        self.log_and_print(f"  Media: {y_mean:,.0f}")
        self.log_and_print(f"  Desviación estándar: {y_std:,.0f}")
        self.log_and_print(f"  Coeficiente de variación: {cv_y:.3f}")
        
        # División train/test simple
        test_size = 0.25
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Aplicar StandardScaler solo a la variable objetivo (y)
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Las variables X se mantienen sin escalar
        X_train_final = X_train.copy()
        X_test_final = X_test.copy()
        
        self.log_and_print(f"Variables X sin escalar - Shape: {X_train_final.shape}")
        self.log_and_print(f"Variable y escalada - Rango train: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
        
        # Validación cruzada simple
        n_splits = 5
        shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        
        self.log_and_print(f"Configuración:")
        self.log_and_print(f"  CV splits: {n_splits}")
        self.log_and_print(f"  División train/test: {len(X_train)}/{len(X_test)} (75%/25%)")
        
        # Espacio de búsqueda simplificado
        search_space = [
            Real(0.1, 100.0, "log-uniform", name='C'),
            Categorical(['linear', 'rbf', 'poly'], name='kernel'),
            Real(0.01, 1.0, "uniform", name='epsilon'),
            Real(1e-4, 1e-1, "log-uniform", name='gamma')
        ]
        
        best_score = -np.inf
        best_params = None
        iteration_count = 0
        cv_scores_history = []
        n_calls = 20
        
        @use_named_args(search_space)
        def objective(**params):
            nonlocal best_score, best_params, iteration_count
            iteration_count += 1
            
            self.log_and_print(f"  Iteración {iteration_count}/{n_calls}: C={params['C']:.3f}, kernel={params['kernel']}, epsilon={params['epsilon']:.3f}")
            
            try:
                svr = SVR(**params)
                scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(shuffle_split.split(X_train_final)):
                    X_train_fold = X_train_final.iloc[train_idx]
                    X_val_fold = X_train_final.iloc[val_idx]
                    y_train_fold = y_train_scaled[train_idx]
                    y_val_fold = y_train_scaled[val_idx]
                    
                    svr.fit(X_train_fold, y_train_fold)
                    
                    # Score en validación
                    y_val_pred = svr.predict(X_val_fold)
                    val_score = r2_score(y_val_fold, y_val_pred)
                    scores.append(val_score)
                
                mean_val_score = np.mean(scores)
                
                self.log_and_print(f"    R² val: {mean_val_score:.4f}")
                
                cv_scores_history.append({
                    'iteration': iteration_count,
                    'val_score': mean_val_score,
                    'params': params.copy()
                })
                
                if mean_val_score > best_score:
                    best_score = mean_val_score
                    best_params = params.copy()
                    self.log_and_print(f"    ✓ Nuevo mejor score: {best_score:.4f}")
                
                return -mean_val_score
                
            except Exception as e:
                self.log_and_print(f"    ✗ Error en iteración: {str(e)}")
                return 1000  # Penalizar fuertemente los errores
        
        self.log_and_print("Iniciando optimización bayesiana...")
        
        # Optimización bayesiana
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=n_calls,
            n_initial_points=5,
            random_state=42,
            acq_func='EI'
        )
        
        self.log_and_print(f"Optimización completada. Mejor score: {best_score:.4f}")
        self.log_and_print(f"Mejores parámetros: {best_params}")
        
        # Entrenar modelo final
        self.log_and_print("Entrenando modelo final...")
        best_model = SVR(**best_params)
        best_model.fit(X_train_final, y_train_scaled)
        
        # Predicciones en escala escalada
        self.log_and_print("Generando predicciones...")
        y_train_pred_scaled = best_model.predict(X_train_final)
        y_test_pred_scaled = best_model.predict(X_test_final)
        
        # Convertir de vuelta a escala original usando el scaler inverso
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
            y_train, y_train_pred, cluster_id, "Train", best_params
        )
        test_plot, test_metrics = self.create_comprehensive_plots(
            y_test, y_test_pred, cluster_id, "Test", best_params
        )
        
        # Gráfico de importancia de features
        importance_plot, feature_importance = self.create_feature_importance_plot(
            best_model, feature_names, cluster_id
        )
        
        # Crear gráfico de evolución del CV
        cv_plot = self.create_cv_evolution_plot(cv_scores_history, cluster_id)
        
        # Guardar modelo y scaler
        self.models[f'cluster_{cluster_id}'] = best_model
        self.scalers[f'cluster_{cluster_id}'] = y_scaler
        
        results = {
            'cluster_id': cluster_id,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_score': best_score,
            'overfitting_ratio': overfitting_ratio,
            'n_features': X.shape[1],
            'n_samples': len(y),
            'cv_history': cv_scores_history,
            'cluster_stats': {
                'mean': y_mean,
                'std': y_std,
                'cv': cv_y,
                'size_category': 'medium'
            },
            'additional_metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mape': train_mape if not np.isnan(train_mape) else 0.0,
                'test_mape': test_mape if not np.isnan(test_mape) else 0.0
            },
            'test_data': {
                'y_true': y_test.tolist(),
                'y_pred': y_test_pred.tolist()
            },
            'train_data': {
                'y_true': y_train.tolist(),
                'y_pred': y_train_pred.tolist()
            },
            'plots': {
                'train': train_plot,
                'test': test_plot,
                'feature_importance': importance_plot,
                'cv_evolution': cv_plot
            },
            'feature_importance': feature_importance
        }
        
        self.log_and_print(f"Modelo para Cluster {cluster_id} completado!")
        
        return results
    
    def create_cv_evolution_plot(self, cv_history, cluster_id):
        """
        Crea un gráfico de la evolución de los scores durante la optimización
        """
        if not cv_history:
            return None
        
        iterations = [h['iteration'] for h in cv_history]
        val_scores = [h['val_score'] for h in cv_history]
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(iterations, val_scores, 'b-', label='Validation R²', alpha=0.7, marker='o')
        plt.xlabel('Iteración')
        plt.ylabel('R² Score')
        plt.title(f'Evolución de Validation R² durante Optimización - Cluster {cluster_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Marcar el mejor score
        best_idx = val_scores.index(max(val_scores))
        plt.axvline(x=iterations[best_idx], color='red', linestyle='--', alpha=0.5, label='Mejor Score')
        plt.scatter(iterations[best_idx], val_scores[best_idx], color='red', s=100, zorder=5)
        
        plt.tight_layout()
        
        plot_filename = os.path.join(self.output_folder, 'graficas', 
                                   f'cv_evolution_cluster_{cluster_id}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_filename
    
    def process_single_file(self, file_path):
        """
        Procesa un solo archivo separando por clusters
        """
        self.log_and_print(f"="*80)
        self.log_and_print(f"PROCESANDO ARCHIVO: {file_path}")
        self.log_and_print(f"="*80)
        
        # Configurar carpeta de salida específica para este archivo
        self.setup_output_folder(file_path)
        
        # Configurar sistema de logging
        self.setup_logging(self.output_folder)
        
        # Cargar datos
        df, available_features, target_column, cluster_column = self.load_and_preprocess_data(file_path)
        
        if df is None:
            self.log_and_print("Error cargando el archivo")
            return
        
        # Información general del dataset
        self.log_and_print(f"\nINFORMACIÓN GENERAL DEL DATASET:")
        self.log_and_print(f"Shape total: {df.shape}")
        self.log_and_print(f"Features disponibles: {len(available_features)}")
        self.log_and_print(f"Variable objetivo: {target_column}")
        self.log_and_print(f"Columna de cluster: {cluster_column}")
        
        # Análisis de clusters
        cluster_counts = df[cluster_column].value_counts().sort_index()
        self.log_and_print(f"\nDISTRIBUCIÓN DE CLUSTERS:")
        for cluster_id, count in cluster_counts.items():
            self.log_and_print(f"  Cluster {cluster_id}: {count} muestras ({count/len(df)*100:.1f}%)")
        
        # Procesar cada cluster
        all_results = {}
        successful_clusters = 0
        
        for cluster_id in sorted(df[cluster_column].unique()):
            self.log_and_print(f"\n{'='*60}")
            self.log_and_print(f"PROCESANDO CLUSTER {cluster_id}")
            self.log_and_print(f"{'='*60}")
            
            try:
                # Filtrar datos del cluster
                cluster_data = df[df[cluster_column] == cluster_id].copy()
                
                # Preprocesar features (solo numéricas)
                self.log_and_print(f"Preprocesando features para Cluster {cluster_id}...")
                processed_features, _ = self.preprocess_features(cluster_data, available_features)
                
                # No guardamos label encoders ya que no hay codificación categórica
                
                # Usar las features procesadas
                X = processed_features.copy()
                y = cluster_data[target_column].copy()
                
                
                # Análisis estadístico del cluster
                y_stats = {
                    'mean': y.mean(),
                    'std': y.std(),
                    'min': y.min(),
                    'max': y.max(),
                    'cv': y.std() / y.mean() if y.mean() > 0 else float('inf')
                }
                
                self.log_and_print(f"Estadísticas del cluster {cluster_id}:")
                self.log_and_print(f"  Media: {y_stats['mean']:,.0f}")
                self.log_and_print(f"  Desv. estándar: {y_stats['std']:,.0f}")
                self.log_and_print(f"  Rango: [{y_stats['min']:,.0f}, {y_stats['max']:,.0f}]")
                self.log_and_print(f"  Coef. variación: {y_stats['cv']:.3f}")
                
                if y_stats['cv'] > 3.0:
                    self.log_and_print(f"  ⚠️  Alta variabilidad detectada en Cluster {cluster_id}")
                elif y_stats['cv'] < 0.1:
                    self.log_and_print(f"  ⚠️  Baja variabilidad detectada en Cluster {cluster_id}")
                else:
                    self.log_and_print(f"  ✓ Variabilidad normal en Cluster {cluster_id}")
                
                # Entrenar modelo
                results = self.train_svr_model_for_cluster(X, y, cluster_id, list(X.columns))
                all_results[cluster_id] = results
                
                # Mostrar resultados
                self.log_and_print(f"\n✓ MODELO ENTRENADO EXITOSAMENTE PARA CLUSTER {cluster_id}")
                self.log_and_print(f"Mejores parámetros: {results['best_params']}")
                self.log_and_print(f"R² Train: {results['train_metrics']['r2']:.4f}")
                self.log_and_print(f"R² Test: {results['test_metrics']['r2']:.4f}")
                self.log_and_print(f"MAE Train: {results['additional_metrics']['train_mae']:,.2f}")
                self.log_and_print(f"MAE Test: {results['additional_metrics']['test_mae']:,.2f}")
                self.log_and_print(f"MSE Train: {results['additional_metrics']['train_mse']:,.2f}")
                self.log_and_print(f"MSE Test: {results['additional_metrics']['test_mse']:,.2f}")
                self.log_and_print(f"RMSE Train: {results['additional_metrics']['train_rmse']:,.2f}")
                self.log_and_print(f"RMSE Test: {results['additional_metrics']['test_rmse']:,.2f}")
                self.log_and_print(f"MAPE Train: {results['additional_metrics']['train_mape']:.2f}%")
                self.log_and_print(f"MAPE Test: {results['additional_metrics']['test_mape']:.2f}%")
                self.log_and_print(f"CV Score: {results['cv_score']:.4f}")
                self.log_and_print(f"Ratio Overfitting: {results['overfitting_ratio']:.2f}")
                
                if results['overfitting_ratio'] > 3.0:
                    self.log_and_print("⚠️  ADVERTENCIA: Sobreajuste severo detectado")
                elif results['overfitting_ratio'] > 2.0:
                    self.log_and_print("⚠️  Sobreajuste moderado detectado")
                else:
                    self.log_and_print("✓ Nivel de sobreajuste aceptable")
                
                successful_clusters += 1
                
            except Exception as e:
                self.log_and_print(f"✗ ERROR procesando cluster {cluster_id}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generar CSV con predicciones para todo el dataset
        if all_results:
            self.log_and_print(f"\n{'='*60}")
            self.log_and_print("GENERANDO CSV CON PREDICCIONES")
            self.log_and_print(f"{'='*60}")
            
            predictions_csv_path, df_with_predictions = self.create_predictions_csv(
                df, available_features, target_column, cluster_column, file_path
            )
        
        # Generar resumen final
        self.generate_final_report(all_results, file_path, cluster_counts)
        
        self.log_and_print(f"\n{'='*80}")
        self.log_and_print(f"PROCESAMIENTO COMPLETADO")
        self.log_and_print(f"Clusters procesados exitosamente: {successful_clusters}")
        self.log_and_print(f"Total de clusters: {len(cluster_counts)}")
        self.log_and_print(f"Resultados guardados en: {self.output_folder}")
        if hasattr(self, 'log_file_path'):
            self.log_and_print(f"Logs completos guardados en: {self.log_file_path}")
        self.log_and_print(f"{'='*80}")

    def generate_final_report(self, results, file_path, cluster_counts):
        """
        Genera un reporte final con todos los resultados
        """
        if not results:
            return
        
        # Crear DataFrame resumen
        summary_data = []
        for cluster_id, result in results.items():
            summary_data.append({
                'Cluster': cluster_id,
                'N_Muestras': result['n_samples'],
                'N_Features': result['n_features'],
                'Train_R2': result['train_metrics']['r2'],
                'Test_R2': result['test_metrics']['r2'],
                'Train_MAE': result['additional_metrics']['train_mae'],
                'Test_MAE': result['additional_metrics']['test_mae'],
                'Train_MSE': result['additional_metrics']['train_mse'],
                'Test_MSE': result['additional_metrics']['test_mse'],
                'Train_RMSE': result['additional_metrics']['train_rmse'],
                'Test_RMSE': result['additional_metrics']['test_rmse'],
                'Train_MAPE': result['additional_metrics']['train_mape'],
                'Test_MAPE': result['additional_metrics']['test_mape'],
                'CV_Score': result['cv_score'],
                'Overfitting_Ratio': result['overfitting_ratio'],
                'Best_Kernel': result['best_params'].get('kernel', 'N/A'),
                'Best_C': result['best_params'].get('C', 'N/A'),
                'Best_Gamma': result['best_params'].get('gamma', 'N/A'),
                'Best_Epsilon': result['best_params'].get('epsilon', 'N/A')
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_R2', ascending=False)
        
        # Guardar resumen
        summary_path = os.path.join(self.output_folder, 'resumen_modelos_por_cluster.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Crear gráfico comparativo
        self.create_comparison_plots(summary_df, results)
        
        # Generar reporte textual
        report_path = os.path.join(self.output_folder, 'reporte_detallado.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DETALLADO - MODELOS SVR POR CLUSTER\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Archivo procesado: {file_path}\n")
            f.write(f"Fecha de procesamiento: {pd.Timestamp.now()}\n\n")
            
            f.write(f"DISTRIBUCIÓN DE CLUSTERS:\n")
            for cluster_id, count in cluster_counts.items():
                f.write(f"  Cluster {cluster_id}: {count} muestras\n")
            f.write("\n")
            
            f.write(f"RESUMEN DE RESULTADOS:\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            f.write(f"ESTADÍSTICAS GENERALES:\n")
            f.write(f"Mejor R² Test: {summary_df['Test_R2'].max():.4f} (Cluster {summary_df.loc[summary_df['Test_R2'].idxmax(), 'Cluster']})\n")
            f.write(f"R² Test promedio: {summary_df['Test_R2'].mean():.4f}\n")
            f.write(f"Desviación estándar R² Test: {summary_df['Test_R2'].std():.4f}\n")
            f.write(f"Clusters procesados exitosamente: {len(summary_df)}\n")
        
        self.log_and_print(f"\nReporte detallado guardado en: {report_path}")
        self.log_and_print(f"Resumen CSV guardado en: {summary_path}")
    
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

    def setup_output_folder(self, file_path):
        """
        Configura la carpeta de salida basada en el nombre del archivo
        """
        # Extraer nombre base del archivo sin extensión
        base_name = os.path.basename(file_path).replace('.csv', '')
        
        # Crear nombre de carpeta de salida
        self.output_folder = f'SVR_{base_name}'
        
        # Crear carpetas
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, 'graficas'), exist_ok=True)
        
        # No usar log_and_print aquí porque el logging aún no está configurado
        print(f"Carpeta de salida configurada: {self.output_folder}")
        
        return self.output_folder

def main():
    if len(sys.argv) != 2:
        print("Uso: python SVR.py <nombre_de_archivo>")
        print("Ejemplo: python SVR.py dataset.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    base_name = filename.replace('.csv', '') if filename.endswith('.csv') else filename
    
    # Directorio de datos
    db_dir = 'DB_separadas'
    
    # Construir ruta del archivo
    file_path = os.path.join(db_dir, f'{base_name}.csv')
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"Error: No se encuentra el archivo {file_path}")
        sys.exit(1)
    
    print(f"Procesando archivo: {file_path}")
    
    # Crear instancia del predictor
    predictor = SVRPredictor()
    
    # Procesar el archivo
    predictor.process_single_file(file_path)
    
    print(f"\n{'='*80}")
    print("PROCESAMIENTO COMPLETADO")
    print(f"Dataset: {base_name}")
    print(f"Resultados guardados en directorio: SVR_{base_name}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()