import pandas as pd
import re
import sys
import os

def procesar_csv(archivo_entrada, archivo_salida):
    """
    Procesa un archivo CSV aplicando dos transformaciones:
    1. En los nombres: convierte guiones en espacios y capitaliza las primeras letras
    2. En los números: formatea a máximo dos decimales
    """
    try:
        # Leer el archivo CSV
        df = pd.read_csv(archivo_entrada)
        
        # Crear una copia para trabajar
        df_procesado = df.copy()
        
        # Procesar cada columna
        for columna in df_procesado.columns:
            # Procesar cada celda de la columna
            for i in range(len(df_procesado)):
                valor = df_procesado.iloc[i, df_procesado.columns.get_loc(columna)]
                
                # Si es un string, aplicar transformaciones de nombre
                if isinstance(valor, str) and not valor.replace('.', '').replace('-', '').replace(',', '').isdigit():
                    # Convertir guiones en espacios
                    valor_procesado = valor.replace('-', ' ')
                    
                    # Capitalizar las primeras letras de cada palabra
                    valor_procesado = ' '.join(word.capitalize() for word in valor_procesado.split())
                    
                    df_procesado.iloc[i, df_procesado.columns.get_loc(columna)] = valor_procesado
                
                # Si es un número, formatear a máximo dos decimales
                elif isinstance(valor, (int, float)) and pd.notna(valor):
                    df_procesado.iloc[i, df_procesado.columns.get_loc(columna)] = round(float(valor), 2)
                
                # Si es un string que representa un número, formatear a máximo dos decimales
                elif isinstance(valor, str) and is_numeric_string(valor):
                    try:
                        numero = float(valor.replace(',', '.'))
                        df_procesado.iloc[i, df_procesado.columns.get_loc(columna)] = round(numero, 2)
                    except ValueError:
                        pass  # Si no se puede convertir, mantener el valor original
        
        # Guardar el archivo procesado
        df_procesado.to_csv(archivo_salida, index=False)
        
        print(f"Archivo procesado guardado como: {archivo_salida}")
        print(f"Filas procesadas: {len(df_procesado)}")
        print(f"Columnas procesadas: {len(df_procesado.columns)}")
        
        return df_procesado
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_entrada}")
        return None
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        return None

def is_numeric_string(s):
    """
    Verifica si un string representa un número
    """
    if not isinstance(s, str):
        return False
    
    # Reemplazar comas por puntos para números decimales
    s_clean = s.replace(',', '.')
    
    # Verificar si es un número (positivo o negativo, con o sin decimales)
    try:
        float(s_clean)
        return True
    except ValueError:
        return False

def mostrar_muestra(df, num_filas=5):
    """
    Muestra una muestra del DataFrame procesado
    """
    if df is not None:
        print("\n--- MUESTRA DEL ARCHIVO PROCESADO ---")
        print(df.head(num_filas))
        print("\n--- INFORMACIÓN DEL ARCHIVO ---")
        print(f"Shape: {df.shape}")
        print(f"Columnas: {list(df.columns)}")

# Función principal
if __name__ == "__main__":
    # Valores por defecto
    archivo_entrada_default = "web/data.csv"
    archivo_salida_default = "web/data_procesado.csv"
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) == 1:
        # Sin argumentos - usar valores por defecto
        archivo_entrada = archivo_entrada_default
        archivo_salida = archivo_salida_default
        print("ℹ️  Usando archivos por defecto")
    elif len(sys.argv) == 2:
        # Solo archivo de entrada especificado
        archivo_entrada = sys.argv[1]
        # Generar nombre de salida basado en el de entrada
        nombre_base = os.path.splitext(archivo_entrada)[0]
        extension = os.path.splitext(archivo_entrada)[1]
        archivo_salida = f"{nombre_base}_procesado{extension}"
    elif len(sys.argv) == 3:
        # Ambos archivos especificados
        archivo_entrada = sys.argv[1]
        archivo_salida = sys.argv[2]
    else:
        # Demasiados argumentos
        print("❌ Error: Demasiados argumentos")
        print("📖 Uso:")
        print("   python procesar_csv.py                           # Usar archivos por defecto")
        print("   python procesar_csv.py <archivo_entrada>         # Especificar solo entrada")
        print("   python procesar_csv.py <archivo_entrada> <archivo_salida>  # Especificar ambos")
        sys.exit(1)
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(archivo_entrada):
        print(f"❌ Error: El archivo de entrada '{archivo_entrada}' no existe")
        sys.exit(1)
    
    print("🔄 Procesando archivo CSV...")
    print(f"📂 Archivo de entrada: {archivo_entrada}")
    print(f"💾 Archivo de salida: {archivo_salida}")
    print("-" * 50)
    
    # Procesar el archivo
    df_resultado = procesar_csv(archivo_entrada, archivo_salida)
    
    # Mostrar una muestra del resultado
    if df_resultado is not None:
        mostrar_muestra(df_resultado)
        
        print("\n✅ Proceso completado exitosamente!")
        print("🎯 Transformaciones aplicadas:")
        print("   • Nombres: guiones → espacios, capitalización")
        print("   • Números: formato con máximo 2 decimales")
        print(f"\n📁 Archivo guardado en: {archivo_salida}")
    else:
        print("❌ Error en el procesamiento")
        sys.exit(1) 