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
        self.model = None
        self.scaler = None
        self.selected_features = []
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

    def get_features_from_dataframe(self, df, target_col='Valor de mercado actual (numérico)'):
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
            if self.model is None:
                self.log_and_print(f"⚠️  Modelo no encontrado")
                return None, None
            
            # Usar las mismas features que se usaron para entrenar
            X = processed_features.copy()
            
            # CORRECCIÓN: NO usar la variable objetivo para predicciones
            # En un escenario real, no tendríamos acceso a los valores reales de y
            # Las variables X NO se escalan (según requerimiento)
            X_for_prediction = X.copy()
            
            # Generar predicciones (el modelo predice en escala escalada)
            y_pred_scaled = self.model.predict(X_for_prediction)
            
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
        plt.title(f'Residuales vs Predichos')
        plt.grid(True, alpha=0.3)
        
        # 3. Histograma de residuales
        ax3 = plt.subplot(3, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Residuales')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de Residuales')
        plt.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax4 = plt.subplot(3, 3, 4)
        stats.probplot(residuals, dist="norm", plot=ax4)
        plt.title(f'Q-Q Plot')
        plt.grid(True, alpha=0.3)
        
        # 5. Boxplot de residuales
        ax5 = plt.subplot(3, 3, 5)
        plt.boxplot(residuals, vert=True)
        plt.ylabel('Residuales')
        plt.title(f'Boxplot Residuales')
        plt.grid(True, alpha=0.3)
        
        # 6. Serie temporal de residuales (orden de predicción)
        ax6 = plt.subplot(3, 3, 6)
        plt.plot(range(len(residuals)), residuals, alpha=0.7, color='purple')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Orden de observación')
        plt.ylabel('Residuales')
        plt.title(f'Serie Temporal Residuales')
        plt.grid(True, alpha=0.3)
        
        # 7. Distribución de valores reales vs predichos
        ax7 = plt.subplot(3, 3, 7)
        plt.hist(y_true, bins=20, alpha=0.5, label='Reales', color='blue')
        plt.hist(y_pred, bins=20, alpha=0.5, label='Predichos', color='red')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución Valores')
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
        plt.title(f'Errores por Percentil')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_filename = os.path.join(self.output_folder, 'graficas', 
                                   f'analisis_completo_{split_name.lower()}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_filename

    def create_feature_importance_plot(self, model, feature_names):
        """
        Crea gráfico de importancia de features
        """
        # Obtener importancias
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Tomar top 20 features
        top_features = feature_importance_df.head(20)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia')
        plt.title('Top 20 Features Más Importantes')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar gráfico
        plot_filename = os.path.join(self.output_folder, 'graficas', 'feature_importance.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_filename, feature_importance_df

    def train_xgboost_model(self, X, y, feature_names):
        """
        Entrena el modelo XGBoost para todo el dataset
        """
        self.log_and_print(f"Entrenando modelo XGBoost...")
        self.log_and_print(f"Shape de X: {X.shape}")
        self.log_and_print(f"Shape de y: {y.shape}")
        
        # Verificar valores nulos
        if X.isnull().sum().sum() > 0:
            self.log_and_print("⚠️  Encontrados valores nulos en X")
            X = X.fillna(X.median())
        
        if y.isnull().sum() > 0:
            self.log_and_print("⚠️  Encontrados valores nulos en y")
            y = y.fillna(y.median())
        
        # División train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        self.log_and_print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        self.log_and_print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
        
        # NO escalar las variables X (según requerimiento)
        X_train_for_model = X_train.copy()
        X_test_for_model = X_test.copy()
        
        # Escalar solo la variable objetivo y
        self.scaler = StandardScaler()
        y_train_scaled = self.scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Configurar parámetros del modelo XGBoost
        model_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.log_and_print(f"Parámetros del modelo: {model_params}")
        
        # Crear y entrenar modelo
        self.model = xgb.XGBRegressor(**model_params)
        
        # Entrenar modelo (con y escalada)
        self.model.fit(X_train_for_model, y_train_scaled)
        
        # Predicciones (en escala escalada)
        y_train_pred_scaled = self.model.predict(X_train_for_model)
        y_test_pred_scaled = self.model.predict(X_test_for_model)
        
        # Convertir de vuelta a escala original
        y_train_pred = self.scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = self.scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calcular métricas
        def safe_mape(y_true, y_pred):
            """Calcula MAPE evitando divisiones por cero"""
            mask = np.abs(y_true) > np.abs(y_true).mean() * 0.01
            if mask.sum() == 0:
                return np.nan
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        
        # Métricas de entrenamiento
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mape = safe_mape(y_train, y_train_pred)
        
        # Métricas de test
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mape = safe_mape(y_test, y_test_pred)
        
        # Ratio de overfitting
        overfitting_ratio = train_rmse / test_rmse if test_rmse > 0 else np.inf
        
        # Log de métricas
        self.log_and_print(f"\n--- MÉTRICAS DE ENTRENAMIENTO ---")
        self.log_and_print(f"R² Train: {train_r2:.4f}")
        self.log_and_print(f"MAE Train: {train_mae:,.2f}")
        self.log_and_print(f"RMSE Train: {train_rmse:,.2f}")
        self.log_and_print(f"MAPE Train: {train_mape:.2f}% {'' if not np.isnan(train_mape) else '(N/A)'}")
        
        self.log_and_print(f"\n--- MÉTRICAS DE TEST ---")
        self.log_and_print(f"R² Test: {test_r2:.4f}")
        self.log_and_print(f"MAE Test: {test_mae:,.2f}")
        self.log_and_print(f"RMSE Test: {test_rmse:,.2f}")
        self.log_and_print(f"MAPE Test: {test_mape:.2f}% {'' if not np.isnan(test_mape) else '(N/A)'}")
        
        self.log_and_print(f"\n--- ANÁLISIS DE OVERFITTING ---")
        self.log_and_print(f"Ratio RMSE (Train/Test): {overfitting_ratio:.2f}")
        if overfitting_ratio > 2.0:
            self.log_and_print("⚠️  Posible overfitting detectado")
        else:
            self.log_and_print("✓ Overfitting controlado")
        
        # Crear gráficos
        self.log_and_print(f"\nGenerando gráficos de análisis...")
        
        # Gráfico de entrenamiento
        train_plot = self.create_comprehensive_plots(
            y_train, y_train_pred, 'Train', model_params
        )
        
        # Gráfico de test
        test_plot = self.create_comprehensive_plots(
            y_test, y_test_pred, 'Test', model_params
        )
        
        # Gráfico de importancia de features
        importance_plot, importance_df = self.create_feature_importance_plot(
            self.model, feature_names
        )
        
        # Guardar features seleccionadas
        self.selected_features = feature_names
        
        # Preparar resultados
        results = {
            'model': self.model,
            'scaler': self.scaler,
            'selected_features': feature_names,
            'metrics': {
                'n_samples': len(X),
                'features_used': len(feature_names),
                'train_r2': train_r2,
                'train_mae': train_mae,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mape': train_mape,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'overfitting_ratio': overfitting_ratio,
                'model_params': model_params
            },
            'plots': {
                'train': train_plot,
                'test': test_plot,
                'importance': importance_plot
            },
            'feature_importance': importance_df
        }
        
        self.log_and_print(f"✓ Modelo entrenado exitosamente")
        
        return results

    def process_single_file(self, file_path):
        """
        Procesa un archivo CSV específico
        """
        self.log_and_print(f"\n{'='*80}")
        self.log_and_print(f"PROCESANDO ARCHIVO: {os.path.basename(file_path)}")
        self.log_and_print(f"{'='*80}")
        
        # Cargar y preprocesar datos
        df, available_features, target_column = self.load_and_preprocess_data(file_path)
        
        if df is None:
            self.log_and_print("Error: No se pudieron cargar los datos")
            return None
        
        self.log_and_print(f"\nDataset completo:")
        self.log_and_print(f"  Total de muestras: {len(df)}")
        
        # Preprocesar features
        processed_features, _ = self.preprocess_features(df, available_features)
        
        if processed_features.empty:
            self.log_and_print(f"⚠️  No hay features válidas para el modelo")
            return None
        
        # Preparar datos para entrenamiento
        X = processed_features
        y = df[target_column]
        
        # Entrenar modelo
        self.log_and_print(f"\n{'-'*60}")
        self.log_and_print(f"ENTRENANDO MODELO XGBOOST")
        self.log_and_print(f"{'-'*60}")
        
        result = self.train_xgboost_model(X, y, processed_features.columns.tolist())
        
        if not result:
            self.log_and_print("✗ Error entrenando el modelo")
            return None
        
        # Generar CSV con predicciones
        self.log_and_print(f"\n{'-'*60}")
        self.log_and_print("GENERANDO ARCHIVO CON PREDICCIONES")
        self.log_and_print(f"{'-'*60}")
        
        predictions_csv, df_with_predictions = self.create_predictions_csv(
            df, available_features, target_column, file_path
        )
        
        return {
            'results': result,
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
        self.output_folder = f'XGBoost_{base_name}_no_clusters'
        
        # Crear carpetas
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, 'graficas'), exist_ok=True)
        
        return self.output_folder

    def generate_final_report(self, results, file_path):
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
        
        # Resumen del modelo
        result = results['results']
        metrics = result['metrics']
        
        self.log_and_print(f"\nRESUMEN DEL MODELO:")
        self.log_and_print(f"{'Métrica':<15} {'Train':<12} {'Test':<12}")
        self.log_and_print(f"{'-'*40}")
        self.log_and_print(f"{'R²':<15} {metrics['train_r2']:<12.4f} {metrics['test_r2']:<12.4f}")
        self.log_and_print(f"{'MAE':<15} {metrics['train_mae']:<12.0f} {metrics['test_mae']:<12.0f}")
        self.log_and_print(f"{'RMSE':<15} {metrics['train_rmse']:<12.0f} {metrics['test_rmse']:<12.0f}")
        if not np.isnan(metrics['train_mape']) and not np.isnan(metrics['test_mape']):
            self.log_and_print(f"{'MAPE (%)':<15} {metrics['train_mape']:<12.2f} {metrics['test_mape']:<12.2f}")
        
        # Crear DataFrame de resumen
        summary_data = {
            'Dataset': base_name,
            'N_Muestras': metrics['n_samples'],
            'N_Features': metrics['features_used'],
            'Train_R2': metrics['train_r2'],
            'Test_R2': metrics['test_r2'],
            'Train_MAE': metrics['train_mae'],
            'Test_MAE': metrics['test_mae'],
            'Train_MSE': metrics['train_mse'],
            'Test_MSE': metrics['test_mse'],
            'Train_RMSE': metrics['train_rmse'],
            'Test_RMSE': metrics['test_rmse'],
            'Train_MAPE': metrics['train_mape'],
            'Test_MAPE': metrics['test_mape'],
            'Overfitting_Ratio': metrics['overfitting_ratio'],
            'N_Estimators': metrics['model_params']['n_estimators'],
            'Max_Depth': metrics['model_params']['max_depth'],
            'Min_Child_Weight': metrics['model_params']['min_child_weight'],
            'Gamma': metrics['model_params']['gamma'],
            'Subsample': metrics['model_params']['subsample'],
            'Colsample_Bytree': metrics['model_params']['colsample_bytree'],
            'Learning_Rate': metrics['model_params']['learning_rate']
        }
        
        summary_df = pd.DataFrame([summary_data])
        
        # Guardar resumen CSV
        summary_csv_path = os.path.join(self.output_folder, 'resumen_modelo.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        self.log_and_print(f"\nResumen CSV guardado en: {summary_csv_path}")
        
        # Estadísticas del modelo
        self.log_and_print(f"\nESTADÍSTICAS DEL MODELO:")
        self.log_and_print(f"R² Test: {metrics['test_r2']:.4f}")
        self.log_and_print(f"RMSE Test: {metrics['test_rmse']:,.0f}")
        self.log_and_print(f"MAE Test: {metrics['test_mae']:,.0f}")
        self.log_and_print(f"Ratio de overfitting: {metrics['overfitting_ratio']:.2f}")
        
        if metrics['overfitting_ratio'] > 2.0:
            self.log_and_print("⚠️  Posible overfitting detectado")
        else:
            self.log_and_print("✓ Overfitting controlado")
        
        # Generar reporte detallado en archivo de texto
        report_path = os.path.join(self.output_folder, 'reporte_detallado.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DETALLADO - MODELO XGBOOST\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Archivo procesado: {file_path}\n")
            f.write(f"Fecha de procesamiento: {pd.Timestamp.now()}\n\n")
            
            f.write(f"CONFIGURACIÓN DEL MODELO:\n")
            f.write(f"Algoritmo: XGBoost Regressor\n")
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
            
            f.write(f"INFORMACIÓN DEL DATASET:\n")
            f.write(f"  Total de muestras: {metrics['n_samples']}\n")
            f.write(f"  Features utilizadas: {metrics['features_used']}\n")
            f.write(f"  División train/test: 80%/20%\n")
            f.write("\n")
            
            f.write(f"RESULTADOS DEL MODELO:\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            f.write(f"MÉTRICAS DETALLADAS:\n")
            f.write(f"ENTRENAMIENTO:\n")
            f.write(f"  R² Score: {metrics['train_r2']:.4f}\n")
            f.write(f"  MAE: {metrics['train_mae']:,.2f}\n")
            f.write(f"  MSE: {metrics['train_mse']:,.2f}\n")
            f.write(f"  RMSE: {metrics['train_rmse']:,.2f}\n")
            if not np.isnan(metrics['train_mape']):
                f.write(f"  MAPE: {metrics['train_mape']:.2f}%\n")
            f.write("\n")
            
            f.write(f"TEST:\n")  
            f.write(f"  R² Score: {metrics['test_r2']:.4f}\n")
            f.write(f"  MAE: {metrics['test_mae']:,.2f}\n")
            f.write(f"  MSE: {metrics['test_mse']:,.2f}\n")
            f.write(f"  RMSE: {metrics['test_rmse']:,.2f}\n")
            if not np.isnan(metrics['test_mape']):
                f.write(f"  MAPE: {metrics['test_mape']:.2f}%\n")
            f.write("\n")
            
            f.write(f"ANÁLISIS DE OVERFITTING:\n")
            f.write(f"Ratio de overfitting (Train RMSE / Test RMSE): {metrics['overfitting_ratio']:.2f}\n")
            f.write(f"Estado: {'⚠️ Alto overfitting' if metrics['overfitting_ratio'] > 2.0 else '✓ Overfitting controlado'}\n")
            f.write("\n")
            
            f.write(f"TOP 10 FEATURES MÁS IMPORTANTES:\n")
            f.write(f"-" * 50 + "\n")
            top_features = result['feature_importance'].head(10)
            for idx, row in top_features.iterrows():
                f.write(f"{row['feature']}: {row['importance']:.4f}\n")
        
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
    Función principal que procesa un dataset específico sin clusters
    """
    # Verificar argumentos de línea de comandos
    if len(sys.argv) != 2:
        print("Uso: python XGBoost_no_clusters.py <ruta_del_archivo>")
        print("Ejemplo: python XGBoost_no_clusters.py DB_viejas/06_db_completo.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
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
    
    predictor.log_and_print(f"Iniciando procesamiento de XGBoost (sin clusters)")
    predictor.log_and_print(f"Archivo: {csv_file}")
    predictor.log_and_print(f"Carpeta de salida: {output_folder}")
    predictor.log_and_print(f"Log file: {log_file}")
    
    try:
        # Procesar archivo
        results = predictor.process_single_file(csv_file)
        
        if results:
            # Generar reporte final
            summary_df = predictor.generate_final_report(results, csv_file)
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