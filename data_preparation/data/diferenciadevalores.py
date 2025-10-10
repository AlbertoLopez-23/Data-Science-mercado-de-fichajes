import pandas as pd

def add_difference_columns(csv_file_path, output_file_path=None):
    """
    Añade dos columnas al CSV: diferencia relativa y diferencia absoluta
    
    Args:
        csv_file_path (str): Ruta del archivo CSV de entrada
        output_file_path (str): Ruta del archivo CSV de salida (opcional)
    """
    # Cargar el CSV
    df = pd.read_csv(csv_file_path)
    
    # Extraer columnas necesarias
    valor_real = df['Valor de mercado actual (numérico)']
    valor_predicho = df['Valor_Predicho']
    
    # Calcular diferencia relativa: (valor_predicho - valor_real) / valor_real * 100
    df['diferencia_relativa'] = ((valor_predicho - valor_real) / valor_real) * 100
    
    # Calcular diferencia absoluta: valor_predicho - valor_real
    df['diferencia_absoluta'] = valor_predicho - valor_real
    
    # Guardar el archivo
    if output_file_path is None:
        output_file_path = csv_file_path.replace('.csv', '_con_diferencias.csv')
    
    df.to_csv(output_file_path, index=False)
    
    print(f"Archivo guardado en: {output_file_path}")
    return df

# Usar la función
if __name__ == "__main__":
    # Cambiar por la ruta de tu archivo
    archivo_entrada = "web/data_preparation/data.csv"  # o "tu_archivo.csv"
    archivo_salida = "resultado.csv"
    
    df = add_difference_columns(archivo_entrada, archivo_salida)