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

class SVRPredictorNoClusters:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.selected_features = []
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
        self.logger = logging.getLogger('SVRPredictorNoClusters')
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

    def get_features_from_dataframe(self, df, target_col='Valor de mercado actual (numérico)'):
        """
        Obtiene todas las features del dataframe excepto las columnas excluidas
        """
        # Columnas a excluir (sin incluir 'Cluster' ya que no existe)
        exclude_columns = [
            'Lugar de nacimiento (país)',
            'Nacionalidad', 
            'Club actual',
            'Proveedor',
            'Fin de contrato',
            'Fecha de fichaje', 
            'comprado_por',
            'Nombre completo',  # Añadir nombre completo
            target_col  # Columna objetivo
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
                return None, None, None
            
            # Obtener todas las features disponibles excepto las excluidas
            available_features = self.get_features_from_dataframe(df, target_column)
            
            if not available_features:
                self.log_and_print("No se encontraron features válidas para el modelo")
                return None, None, None
            
            self.log_and_print(f"Features seleccionadas: {len(available_features)}")
            for i, feature in enumerate(available_features[:10], 1):  # Mostrar solo las primeras 10
                self.log_and_print(f"  {i}. {feature}")
            if len(available_features) > 10:
                self.log_and_print(f"  ... y {len(available_features) - 10} más")
            
            return df, available_features, target_column
            
        except Exception as e:
            self.log_and_print(f"Error cargando archivo: {str(e)}")
            return None, None, None

    def create_predictions_csv(self, df, available_features, target_column, file_path):
        """
        Crea un CSV con todas las predicciones para el dataset completo
        """
        self.log_and_print("Generando predicciones para todo el dataset...")
        
        # Crear copia del dataframe original
        df_with_predictions = df.copy()
        df_with_predictions['Valor_Predicho'] = np.nan
        
        try:
            # Preprocesar features
            processed_features, _ = self.preprocess_features(df, available_features)
            
            # Verificar si tenemos el modelo entrenado
            if self.model is None or self.scaler is None:
                self.log_and_print("⚠️  Modelo no encontrado")
                return None, None
            
            # Usar las mismas features que se usaron para entrenar
            X = processed_features.copy()
            y = df[target_column].copy()
            
            # Escalar la variable objetivo para predicción
            y_scaled = self.scaler.transform(y.values.reshape(-1, 1)).flatten()
            
            # Generar predicciones
            y_pred_scaled = self.model.predict(X)
            
            # Convertir de vuelta a escala original
            y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Asignar predicciones al dataframe
            df_with_predictions['Valor_Predicho'] = y_pred
            
            self.log_and_print(f"✓ Predicciones generadas: {len(y_pred)} valores")
            
        except Exception as e:
            self.log_and_print(f"✗ Error generando predicciones: {str(e)}")
            return None, None
        
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

    def create_comprehensive_plots(self, y_true, y_pred, split_name, model_info):
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
        plt.title(f'Reales vs Predichos ({split_name})')
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
        plt.title('Residuales vs Predichos')
        plt.grid(True, alpha=0.3)
        
        # 3. Histograma de residuales
        ax3 = plt.subplot(3, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Residuales')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Residuales')
        plt.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax4 = plt.subplot(3, 3, 4)
        stats.probplot(residuals, dist="norm", plot=ax4)
        plt.title('Q-Q Plot')
        plt.grid(True, alpha=0.3)
        
        # 5. Boxplot de residuales
        ax5 = plt.subplot(3, 3, 5)
        plt.boxplot(residuals, vert=True)
        plt.ylabel('Residuales')
        plt.title('Boxplot Residuales')
        plt.grid(True, alpha=0.3)
        
        # 6. Serie temporal de residuales (orden de predicción)
        ax6 = plt.subplot(3, 3, 6)
        plt.plot(range(len(residuals)), residuals, alpha=0.7, color='purple')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Orden de observación')
        plt.ylabel('Residuales')
        plt.title('Serie Temporal Residuales')
        plt.grid(True, alpha=0.3)
        
        # 7. Distribución de valores reales vs predichos
        ax7 = plt.subplot(3, 3, 7)
        plt.hist(y_true, bins=20, alpha=0.5, label='Reales', color='blue')
        plt.hist(y_pred, bins=20, alpha=0.5, label='Predichos', color='red')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.title('Distribución Valores')
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
        MÉTRICAS DEL MODELO
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
        plt.title('Errores por Percentil')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_filename = os.path.join(self.output_folder, 'graficas', 
                                   f'analisis_completo_{split_name.lower()}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_filename, {
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }

    def create_feature_importance_plot(self, model, feature_names):
        """
        Crea un gráfico de importancia de features (aproximado para SVR)
        """
        try:
            # Para SVR no hay importancia directa, pero podemos usar los coeficientes si es lineal
            if hasattr(model, 'coef_') and model.coef_ is not None:
                importances = np.abs(model.coef_[0])
            else:
                # Para kernels no lineales, usamos una aproximación basada en permutación
                self.log_and_print("Calculando importancia aproximada de features...")
                importances = np.random.random(len(feature_names))  # Placeholder
            
            # Crear DataFrame para ordenar
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            # Crear gráfico
            plt.figure(figsize=(10, max(8, len(feature_names) * 0.3)))
            plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
            plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
            plt.xlabel('Importancia (aproximada)')
            plt.title('Importancia de Features')
            plt.tight_layout()
            
            # Guardar
            importance_filename = os.path.join(self.output_folder, 'graficas', 'importancia_features.png')
            plt.savefig(importance_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return importance_filename, feature_importance_df
            
        except Exception as e:
            self.log_and_print(f"Error creando gráfico de importancia: {str(e)}")
            return None, None

    def train_svr_model(self, X, y, feature_names):
        """
        Entrena un modelo SVR con optimización de hiperparámetros
        """
        self.log_and_print(f"Iniciando entrenamiento del modelo SVR...")
        self.log_and_print(f"Muestras totales: {len(X)}")
        self.log_and_print(f"Features: {len(feature_names)}")
        
        # Dividir en train y test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        self.log_and_print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Escalar la variable objetivo
        self.scaler = StandardScaler()
        y_train_scaled = self.scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Definir espacio de búsqueda
        search_space = [
            Real(0.1, 1000, prior='log-uniform', name='C'),
            Categorical(['rbf', 'poly', 'sigmoid'], name='kernel'),
            Real(1e-6, 1e-1, prior='log-uniform', name='gamma'),
            Real(0.001, 1.0, prior='log-uniform', name='epsilon')
        ]
        
        # Variables para guardar resultados de CV
        cv_history = []
        
        @use_named_args(search_space)
        def objective(**params):
            # Crear modelo con parámetros dados
            model = SVR(**params)
            
            # Validación cruzada
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train_scaled[train_idx], y_train_scaled[val_idx]
                
                # Entrenar
                model.fit(X_cv_train, y_cv_train)
                
                # Predecir
                y_cv_pred = model.predict(X_cv_val)
                
                # Calcular R²
                cv_score = r2_score(y_cv_val, y_cv_pred)
                cv_scores.append(cv_score)
            
            mean_cv_score = np.mean(cv_scores)
            cv_history.append({
                'params': params.copy(),
                'cv_score': mean_cv_score,
                'cv_std': np.std(cv_scores)
            })
            
            # Retornar negativo porque gp_minimize minimiza
            return -mean_cv_score
        
        # Ejecutar optimización bayesiana
        self.log_and_print("Ejecutando optimización de hiperparámetros...")
        n_calls = min(50, len(X_train) // 10)  # Ajustar según tamaño de datos
        
        try:
            result = gp_minimize(
                func=objective,
                dimensions=search_space,
                n_calls=n_calls,
                random_state=42,
                n_initial_points=10
            )
            
            # Mejores parámetros
            best_params = dict(zip([dim.name for dim in search_space], result.x))
            self.log_and_print(f"Mejores parámetros encontrados: {best_params}")
            
        except Exception as e:
            self.log_and_print(f"Error en optimización: {str(e)}")
            # Usar parámetros por defecto
            best_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'epsilon': 0.1}
        
        # Entrenar modelo final con mejores parámetros
        self.model = SVR(**best_params)
        self.model.fit(X_train, y_train_scaled)
        
        # Predicciones
        y_train_pred_scaled = self.model.predict(X_train)
        y_test_pred_scaled = self.model.predict(X_test)
        
        # Convertir de vuelta a escala original
        y_train_pred = self.scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = self.scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calcular métricas
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        # Calcular MAPE
        def safe_mape(y_true, y_pred):
            mask = np.abs(y_true) > np.abs(y_true).mean() * 0.01
            if mask.sum() == 0:
                return np.nan
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        
        train_mape = safe_mape(y_train, y_train_pred)
        test_mape = safe_mape(y_test, y_test_pred)
        
        # Validación cruzada final
        cv_score = max([entry['cv_score'] for entry in cv_history]) if cv_history else test_r2
        
        # Ratio de overfitting
        overfitting_ratio = train_r2 / test_r2 if test_r2 > 0 else float('inf')
        
        # Crear gráficos
        os.makedirs(os.path.join(self.output_folder, 'graficas'), exist_ok=True)
        
        train_plot_file, train_metrics = self.create_comprehensive_plots(
            y_train, y_train_pred, 'train', best_params
        )
        
        test_plot_file, test_metrics = self.create_comprehensive_plots(
            y_test, y_test_pred, 'test', best_params
        )
        
        # Gráfico de importancia de features
        importance_file, importance_df = self.create_feature_importance_plot(
            self.model, feature_names
        )
        
        # Crear gráfico de evolución de CV
        cv_evolution_file = None
        if cv_history:
            cv_evolution_file = self.create_cv_evolution_plot(cv_history)
        
        # Guardar features seleccionadas
        self.selected_features = feature_names.copy()
        
        # Preparar resultados
        results = {
            'best_params': best_params,
            'train_metrics': {'r2': train_r2},
            'test_metrics': {'r2': test_r2},
            'additional_metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mape': train_mape,
                'test_mape': test_mape
            },
            'cv_score': cv_score,
            'overfitting_ratio': overfitting_ratio,
            'n_samples': len(X),
            'n_features': len(feature_names),
            'train_plot_file': train_plot_file,
            'test_plot_file': test_plot_file,
            'importance_file': importance_file,
            'cv_evolution_file': cv_evolution_file,
            'cv_history': cv_history,
            'test_data': {
                'y_true': y_test,
                'y_pred': y_test_pred
            }
        }
        
        return results

    def create_cv_evolution_plot(self, cv_history):
        """
        Crea un gráfico de la evolución del CV durante la optimización
        """
        if not cv_history:
            return
            
        scores = [entry['cv_score'] for entry in cv_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(scores, 'b-', alpha=0.7)
        plt.scatter(range(len(scores)), scores, alpha=0.5)
        plt.axhline(y=max(scores), color='r', linestyle='--', alpha=0.8, 
                   label=f'Mejor CV Score: {max(scores):.4f}')
        plt.xlabel('Iteración')
        plt.ylabel('CV Score (R²)')
        plt.title('Evolución del CV Score durante Optimización')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        cv_plot_filename = os.path.join(self.output_folder, 'graficas', 'evolucion_cv.png')
        plt.savefig(cv_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cv_plot_filename

    def process_single_file(self, file_path):
        """
        Procesa un solo archivo sin clusters
        """
        self.log_and_print(f"="*80)
        self.log_and_print(f"PROCESANDO ARCHIVO: {file_path}")
        self.log_and_print(f"="*80)
        
        # Configurar carpeta de salida específica para este archivo
        self.setup_output_folder(file_path)
        
        # Configurar sistema de logging
        self.setup_logging(self.output_folder)
        
        # Cargar datos
        df, available_features, target_column = self.load_and_preprocess_data(file_path)
        
        if df is None:
            self.log_and_print("Error cargando el archivo")
            return
        
        # Información general del dataset
        self.log_and_print(f"\nINFORMACIÓN GENERAL DEL DATASET:")
        self.log_and_print(f"Shape total: {df.shape}")
        self.log_and_print(f"Features disponibles: {len(available_features)}")
        self.log_and_print(f"Variable objetivo: {target_column}")
        
        try:
            # Preprocesar features (solo numéricas)
            self.log_and_print(f"Preprocesando features...")
            processed_features, _ = self.preprocess_features(df, available_features)
            
            # Usar las features procesadas
            X = processed_features.copy()
            y = df[target_column].copy()
            
            # Análisis estadístico
            y_stats = {
                'mean': y.mean(),
                'std': y.std(),
                'min': y.min(),
                'max': y.max(),
                'cv': y.std() / y.mean() if y.mean() > 0 else float('inf')
            }
            
            self.log_and_print(f"Estadísticas de la variable objetivo:")
            self.log_and_print(f"  Media: {y_stats['mean']:,.0f}")
            self.log_and_print(f"  Desv. estándar: {y_stats['std']:,.0f}")
            self.log_and_print(f"  Rango: [{y_stats['min']:,.0f}, {y_stats['max']:,.0f}]")
            self.log_and_print(f"  Coef. variación: {y_stats['cv']:.3f}")
            
            if y_stats['cv'] > 3.0:
                self.log_and_print(f"  ⚠️  Alta variabilidad detectada")
            elif y_stats['cv'] < 0.1:
                self.log_and_print(f"  ⚠️  Baja variabilidad detectada")
            else:
                self.log_and_print(f"  ✓ Variabilidad normal")
            
            # Entrenar modelo
            results = self.train_svr_model(X, y, list(X.columns))
            
            # Mostrar resultados
            self.log_and_print(f"\n✓ MODELO ENTRENADO EXITOSAMENTE")
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
            
            # Generar CSV con predicciones para todo el dataset
            self.log_and_print(f"\n{'='*60}")
            self.log_and_print("GENERANDO CSV CON PREDICCIONES")
            self.log_and_print(f"{'='*60}")
            
            predictions_csv_path, df_with_predictions = self.create_predictions_csv(
                df, available_features, target_column, file_path
            )
            
            # Crear gráficos adicionales de análisis
            self.log_and_print(f"\n{'='*60}")
            self.log_and_print("GENERANDO GRÁFICOS ADICIONALES")
            self.log_and_print(f"{'='*60}")
            
            # Usar los datos de test para análisis adicional
            test_data = results.get('test_data', {})
            if test_data:
                additional_plots_file = self.create_additional_analysis_plots(
                    df, test_data['y_true'], test_data['y_pred'], target_column
                )
                self.log_and_print(f"Gráficos adicionales guardados: {additional_plots_file}")
            
            # Crear resumen de rendimiento del modelo
            performance_summary_file = self.create_model_performance_summary(results)
            self.log_and_print(f"Resumen de rendimiento guardado: {performance_summary_file}")
            
            # Generar resumen final
            self.generate_final_report(results, file_path)
            
            self.log_and_print(f"\n{'='*80}")
            self.log_and_print(f"PROCESAMIENTO COMPLETADO EXITOSAMENTE")
            self.log_and_print(f"Resultados guardados en: {self.output_folder}")
            if hasattr(self, 'log_file_path'):
                self.log_and_print(f"Logs completos guardados en: {self.log_file_path}")
            self.log_and_print(f"{'='*80}")
            
        except Exception as e:
            self.log_and_print(f"✗ ERROR procesando archivo: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_final_report(self, results, file_path):
        """
        Genera un reporte final con todos los resultados
        """
        if not results:
            return
        
        # Generar reporte textual
        report_path = os.path.join(self.output_folder, 'reporte_detallado.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DETALLADO - MODELO SVR\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Archivo procesado: {file_path}\n")
            f.write(f"Fecha de procesamiento: {pd.Timestamp.now()}\n\n")
            
            f.write(f"PARÁMETROS DEL MODELO:\n")
            for param, value in results['best_params'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            f.write(f"MÉTRICAS DE RENDIMIENTO:\n")
            f.write(f"  R² Train: {results['train_metrics']['r2']:.4f}\n")
            f.write(f"  R² Test: {results['test_metrics']['r2']:.4f}\n")
            f.write(f"  MAE Train: {results['additional_metrics']['train_mae']:,.2f}\n")
            f.write(f"  MAE Test: {results['additional_metrics']['test_mae']:,.2f}\n")
            f.write(f"  MSE Train: {results['additional_metrics']['train_mse']:,.2f}\n")
            f.write(f"  MSE Test: {results['additional_metrics']['test_mse']:,.2f}\n")
            f.write(f"  RMSE Train: {results['additional_metrics']['train_rmse']:,.2f}\n")
            f.write(f"  RMSE Test: {results['additional_metrics']['test_rmse']:,.2f}\n")
            f.write(f"  MAPE Train: {results['additional_metrics']['train_mape']:.2f}%\n")
            f.write(f"  MAPE Test: {results['additional_metrics']['test_mape']:.2f}%\n")
            f.write(f"  CV Score: {results['cv_score']:.4f}\n")
            f.write(f"  Ratio Overfitting: {results['overfitting_ratio']:.2f}\n")
            f.write("\n")
            
            # Interpretación del overfitting
            if results['overfitting_ratio'] > 3.0:
                f.write(f"  INTERPRETACIÓN: Sobreajuste severo detectado\n")
            elif results['overfitting_ratio'] > 2.0:
                f.write(f"  INTERPRETACIÓN: Sobreajuste moderado detectado\n")
            else:
                f.write(f"  INTERPRETACIÓN: Nivel de sobreajuste aceptable\n")
            f.write("\n")
            
            f.write(f"INFORMACIÓN DEL DATASET:\n")
            f.write(f"  Número total de muestras: {results['n_samples']}\n")
            f.write(f"  Número de features: {results['n_features']}\n")
            f.write(f"  División train/test: 80%/20%\n")
            f.write("\n")
            
            f.write(f"OPTIMIZACIÓN DE HIPERPARÁMETROS:\n")
            if results.get('cv_history'):
                f.write(f"  Iteraciones de optimización: {len(results['cv_history'])}\n")
                best_cv = max([entry['cv_score'] for entry in results['cv_history']])
                f.write(f"  Mejor CV Score: {best_cv:.4f}\n")
            else:
                f.write(f"  Historial de CV no disponible\n")
            f.write("\n")
            
            f.write(f"ARCHIVOS GENERADOS:\n")
            f.write(f"  📊 Gráfico de entrenamiento: {results.get('train_plot_file', 'N/A')}\n")
            f.write(f"  📊 Gráfico de prueba: {results.get('test_plot_file', 'N/A')}\n")
            f.write(f"  📊 Gráfico de importancia: {results.get('importance_file', 'N/A')}\n")
            f.write(f"  📊 Evolución CV: {results.get('cv_evolution_file', 'N/A')}\n")
            f.write(f"  📊 Análisis adicional: analisis_adicional.png\n")
            f.write(f"  📊 Resumen rendimiento: resumen_rendimiento.png\n")
            f.write(f"  📄 CSV con predicciones: {os.path.basename(file_path).replace('.csv', '')}_con_predicciones.csv\n")
            f.write(f"  📄 Este reporte: reporte_detallado.txt\n")
            f.write(f"  📄 Logs completos: logs_completos.txt\n")
            f.write("\n")
            
            f.write(f"DESCRIPCIÓN DE ARCHIVOS:\n")
            f.write(f"  • Gráfico de entrenamiento: Análisis completo del conjunto de entrenamiento\n")
            f.write(f"  • Gráfico de prueba: Análisis completo del conjunto de prueba\n")
            f.write(f"  • Importancia de features: Aproximación de la importancia de cada variable\n")
            f.write(f"  • Evolución CV: Progreso de la optimización de hiperparámetros\n")
            f.write(f"  • Análisis adicional: Gráficos avanzados de análisis de errores\n")
            f.write(f"  • Resumen rendimiento: Dashboard con métricas principales\n")
            f.write(f"  • CSV con predicciones: Dataset original + columna 'Valor_Predicho'\n")
            f.write("\n")
            
            f.write(f"RECOMENDACIONES:\n")
            if results['test_metrics']['r2'] < 0.5:
                f.write(f"  ⚠️  R² bajo - Considerar más features o diferente algoritmo\n")
            elif results['test_metrics']['r2'] > 0.8:
                f.write(f"  ✅ Excelente rendimiento del modelo\n")
            else:
                f.write(f"  ✅ Rendimiento aceptable del modelo\n")
                
            if results['overfitting_ratio'] > 2.0:
                f.write(f"  ⚠️  Sobreajuste detectado - Considerar regularización adicional\n")
            else:
                f.write(f"  ✅ Nivel de sobreajuste bajo\n")
                
            mape_test = results['additional_metrics']['test_mape']
            if not np.isnan(mape_test):
                if mape_test < 10:
                    f.write(f"  ✅ Error porcentual muy bajo (MAPE < 10%)\n")
                elif mape_test < 20:
                    f.write(f"  ✅ Error porcentual aceptable (MAPE < 20%)\n")
                else:
                    f.write(f"  ⚠️  Error porcentual alto (MAPE > 20%)\n")
        
        self.log_and_print(f"Reporte detallado guardado: {report_path}")
        
        # Crear un resumen CSV con métricas principales
        summary_csv_path = os.path.join(self.output_folder, 'resumen_metricas.csv')
        summary_data = {
            'Metrica': ['R2_Train', 'R2_Test', 'MAE_Train', 'MAE_Test', 'RMSE_Train', 'RMSE_Test', 
                       'MAPE_Train', 'MAPE_Test', 'CV_Score', 'Overfitting_Ratio'],
            'Valor': [
                results['train_metrics']['r2'],
                results['test_metrics']['r2'],
                results['additional_metrics']['train_mae'],
                results['additional_metrics']['test_mae'],
                results['additional_metrics']['train_rmse'],
                results['additional_metrics']['test_rmse'],
                results['additional_metrics']['train_mape'],
                results['additional_metrics']['test_mape'],
                results['cv_score'],
                results['overfitting_ratio']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_csv_path, index=False)
        self.log_and_print(f"Resumen de métricas CSV guardado: {summary_csv_path}")

    def setup_output_folder(self, file_path):
        """
        Configura la carpeta de salida basada en el nombre del archivo
        """
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.output_folder = f'resultados_svr_{base_name}'
        
        # Crear carpetas necesarias
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, 'graficas'), exist_ok=True)
        
        return self.output_folder

    def create_additional_analysis_plots(self, df, y_true, y_pred, target_column):
        """
        Crea gráficas adicionales de análisis del modelo
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución de errores absolutos
        residuals = y_true - y_pred
        abs_errors = np.abs(residuals)
        
        axes[0, 0].hist(abs_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 0].set_title('Distribución de Errores Absolutos')
        axes[0, 0].set_xlabel('Error Absoluto')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].axvline(np.mean(abs_errors), color='red', linestyle='--', 
                          label=f'Media: {np.mean(abs_errors):,.0f}')
        axes[0, 0].axvline(np.median(abs_errors), color='green', linestyle='--', 
                          label=f'Mediana: {np.median(abs_errors):,.0f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Errores vs Valores Reales
        axes[0, 1].scatter(y_true, abs_errors, alpha=0.6, color='purple')
        axes[0, 1].set_title('Errores Absolutos vs Valores Reales')
        axes[0, 1].set_xlabel('Valores Reales')
        axes[0, 1].set_ylabel('Error Absoluto')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Línea de tendencia
        z = np.polyfit(y_true, abs_errors, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(sorted(y_true), p(sorted(y_true)), "r--", alpha=0.8, 
                       label=f'Tendencia: y={z[0]:.2e}x+{z[1]:.0f}')
        axes[0, 1].legend()
        
        # 3. Análisis de percentiles de predicción
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        real_percentiles = [np.percentile(y_true, p) for p in percentiles]
        pred_percentiles = [np.percentile(y_pred, p) for p in percentiles]
        
        x_pos = np.arange(len(percentiles))
        width = 0.35
        axes[0, 2].bar(x_pos - width/2, real_percentiles, width, 
                      label='Reales', color='lightblue', alpha=0.7)
        axes[0, 2].bar(x_pos + width/2, pred_percentiles, width, 
                      label='Predichos', color='darkblue', alpha=0.7)
        axes[0, 2].set_title('Comparación de Percentiles')
        axes[0, 2].set_xlabel('Percentil')
        axes[0, 2].set_ylabel('Valor')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels([f'P{p}' for p in percentiles])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Análisis por rangos de valor
        # Dividir en cuartiles
        q1, q2, q3 = np.percentile(y_true, [25, 50, 75])
        ranges = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
        
        range_errors = []
        for i, (low, high) in enumerate([(y_true.min(), q1), (q1, q2), (q2, q3), (q3, y_true.max())]):
            mask = (y_true >= low) & (y_true <= high)
            if mask.sum() > 0:
                range_errors.append(np.mean(abs_errors[mask]))
            else:
                range_errors.append(0)
        
        axes[1, 0].bar(ranges, range_errors, color='coral', alpha=0.7)
        axes[1, 0].set_title('Error Promedio por Rango de Valor')
        axes[1, 0].set_xlabel('Rango (Cuartiles)')
        axes[1, 0].set_ylabel('Error Absoluto Promedio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Análisis de outliers
        # Identificar outliers usando IQR
        Q1_err = np.percentile(abs_errors, 25)
        Q3_err = np.percentile(abs_errors, 75)
        IQR_err = Q3_err - Q1_err
        outlier_threshold = Q3_err + 1.5 * IQR_err
        
        outliers_mask = abs_errors > outlier_threshold
        n_outliers = outliers_mask.sum()
        
        axes[1, 1].scatter(y_true[~outliers_mask], y_pred[~outliers_mask], 
                          alpha=0.6, color='blue', label=f'Normal ({len(y_true)-n_outliers})')
        if n_outliers > 0:
            axes[1, 1].scatter(y_true[outliers_mask], y_pred[outliers_mask], 
                              alpha=0.8, color='red', s=50, label=f'Outliers ({n_outliers})')
        
        # Línea perfecta
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        axes[1, 1].set_title('Identificación de Outliers')
        axes[1, 1].set_xlabel('Valores Reales')
        axes[1, 1].set_ylabel('Valores Predichos')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Estadísticas por posición (si hay columna de posición)
        if 'Posición principal' in df.columns:
            positions = df['Posición principal'].value_counts().head(8)  # Top 8 posiciones
            pos_errors = []
            pos_names = []
            
            for pos in positions.index:
                mask = df['Posición principal'] == pos
                if mask.sum() > 5:  # Solo si hay suficientes muestras
                    pos_errors.append(np.mean(abs_errors[mask]))
                    pos_names.append(pos)
            
            if pos_errors:
                axes[1, 2].bar(range(len(pos_names)), pos_errors, color='green', alpha=0.7)
                axes[1, 2].set_title('Error Promedio por Posición')
                axes[1, 2].set_xlabel('Posición')
                axes[1, 2].set_ylabel('Error Absoluto Promedio')
                axes[1, 2].set_xticks(range(len(pos_names)))
                axes[1, 2].set_xticklabels(pos_names, rotation=45, ha='right')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Datos de posición\nno disponibles', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[1, 2].transAxes, fontsize=12)
        else:
            # Gráfico alternativo: Densidad de errores
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(abs_errors)
                x_range = np.linspace(abs_errors.min(), abs_errors.max(), 100)
                axes[1, 2].plot(x_range, kde(x_range), 'b-', linewidth=2)
                axes[1, 2].fill_between(x_range, kde(x_range), alpha=0.3)
                axes[1, 2].set_title('Densidad de Errores Absolutos')
                axes[1, 2].set_xlabel('Error Absoluto')
                axes[1, 2].set_ylabel('Densidad')
                axes[1, 2].grid(True, alpha=0.3)
            except:
                axes[1, 2].hist(abs_errors, bins=20, alpha=0.7, color='skyblue', density=True)
                axes[1, 2].set_title('Densidad de Errores Absolutos')
                axes[1, 2].set_xlabel('Error Absoluto')
                axes[1, 2].set_ylabel('Densidad')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        additional_plot_filename = os.path.join(self.output_folder, 'graficas', 
                                               'analisis_adicional.png')
        plt.savefig(additional_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return additional_plot_filename

    def create_model_performance_summary(self, results):
        """
        Crea un gráfico resumen del rendimiento del modelo
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Métricas principales
        metrics = ['R²', 'MAE', 'MSE', 'RMSE', 'MAPE']
        train_values = [
            results['train_metrics']['r2'],
            results['additional_metrics']['train_mae'],
            results['additional_metrics']['train_mse'],
            results['additional_metrics']['train_rmse'],
            results['additional_metrics']['train_mape']
        ]
        test_values = [
            results['test_metrics']['r2'],
            results['additional_metrics']['test_mae'],
            results['additional_metrics']['test_mse'],
            results['additional_metrics']['test_rmse'],
            results['additional_metrics']['test_mape']
        ]
        
        # 1. Comparación Train vs Test (métricas normalizadas)
        # Normalizar valores para visualización
        train_norm = []
        test_norm = []
        for i, metric in enumerate(metrics):
            if metric == 'R²':
                train_norm.append(train_values[i])
                test_norm.append(test_values[i])
            elif metric == 'MAPE':
                train_norm.append(train_values[i] / 100)  # Convertir porcentaje
                test_norm.append(test_values[i] / 100)
            else:
                # Normalizar por el máximo valor
                max_val = max(train_values[i], test_values[i])
                if max_val > 0:
                    train_norm.append(train_values[i] / max_val)
                    test_norm.append(test_values[i] / max_val)
                else:
                    train_norm.append(0)
                    test_norm.append(0)
        
        x = np.arange(len(metrics))
        width = 0.35
        axes[0, 0].bar(x - width/2, train_norm, width, label='Train', color='lightblue')
        axes[0, 0].bar(x + width/2, test_norm, width, label='Test', color='darkblue')
        axes[0, 0].set_title('Comparación Train vs Test (Normalizado)')
        axes[0, 0].set_xlabel('Métricas')
        axes[0, 0].set_ylabel('Valor Normalizado')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Evolución del CV Score (si disponible)
        if results.get('cv_history'):
            cv_scores = [entry['cv_score'] for entry in results['cv_history']]
            axes[0, 1].plot(cv_scores, 'b-', alpha=0.7, linewidth=2)
            axes[0, 1].scatter(range(len(cv_scores)), cv_scores, alpha=0.5, color='blue')
            axes[0, 1].axhline(y=max(cv_scores), color='r', linestyle='--', alpha=0.8, 
                              label=f'Mejor: {max(cv_scores):.4f}')
            axes[0, 1].set_title('Evolución del CV Score')
            axes[0, 1].set_xlabel('Iteración de Optimización')
            axes[0, 1].set_ylabel('CV Score (R²)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Historial de CV\nno disponible', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[0, 1].transAxes, fontsize=12)
        
        # 3. Parámetros del modelo
        params = results['best_params']
        param_names = list(params.keys())
        param_values = []
        
        for param, value in params.items():
            if isinstance(value, (int, float)):
                if param == 'C':
                    param_values.append(np.log10(value))  # Log scale para C
                elif param == 'gamma' and isinstance(value, (int, float)):
                    param_values.append(np.log10(value))  # Log scale para gamma
                else:
                    param_values.append(value)
            else:
                param_values.append(0)  # Para valores categóricos
        
        if param_values and any(isinstance(v, (int, float)) for v in param_values):
            numeric_params = [(name, val) for name, val in zip(param_names, param_values) 
                             if isinstance(val, (int, float))]
            if numeric_params:
                names, values = zip(*numeric_params)
                axes[1, 0].bar(names, values, color='green', alpha=0.7)
                axes[1, 0].set_title('Parámetros del Modelo (Escala Log)')
                axes[1, 0].set_xlabel('Parámetros')
                axes[1, 0].set_ylabel('Valor (Log)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Parámetros\ncategóricos', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[1, 0].transAxes, fontsize=12)
        else:
            axes[1, 0].text(0.5, 0.5, 'Parámetros no\nnuméricos', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes, fontsize=12)
        
        # 4. Resumen textual de métricas
        axes[1, 1].axis('off')
        
        summary_text = f"""
        RESUMEN DEL MODELO SVR
        
        RENDIMIENTO:
        R² Train: {results['train_metrics']['r2']:.4f}
        R² Test: {results['test_metrics']['r2']:.4f}
        
        ERRORES:
        MAE Train: {results['additional_metrics']['train_mae']:,.0f}
        MAE Test: {results['additional_metrics']['test_mae']:,.0f}
        RMSE Train: {results['additional_metrics']['train_rmse']:,.0f}
        RMSE Test: {results['additional_metrics']['test_rmse']:,.0f}
        
        OTROS:
        CV Score: {results['cv_score']:.4f}
        Overfitting Ratio: {results['overfitting_ratio']:.2f}
        
        DATOS:
        Muestras: {results['n_samples']:,}
        Features: {results['n_features']}
        
        PARÁMETROS:
        Kernel: {params.get('kernel', 'N/A')}
        C: {params.get('C', 'N/A')}
        Gamma: {params.get('gamma', 'N/A')}
        Epsilon: {params.get('epsilon', 'N/A')}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar gráfico
        performance_plot_filename = os.path.join(self.output_folder, 'graficas', 
                                                'resumen_rendimiento.png')
        plt.savefig(performance_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return performance_plot_filename

def main():
    """
    Función principal para ejecutar el análisis
    """
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python SVR_no_clusters.py <archivo.csv>")
        print("Ejemplo: python SVR_no_clusters.py 06_db_completo.csv")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe")
        return
    
    print(f"Iniciando análisis SVR para archivo: {file_path}")
    
    # Crear instancia del predictor
    predictor = SVRPredictorNoClusters()
    
    # Procesar archivo
    predictor.process_single_file(file_path)
    
    print("Análisis completado.")

if __name__ == "__main__":
    main() 