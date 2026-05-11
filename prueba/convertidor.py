import os

def cambiar_valor_en_txt(carpeta, nuevo_valor):
    # Recorre todos los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.txt'):
            ruta_archivo = os.path.join(carpeta, archivo)
            with open(ruta_archivo, 'r') as file:
                lineas = file.readlines()
            
            # Cambiar el primer valor de cada línea
            lineas_modificadas = []
            for linea in lineas:
                partes = linea.split()
                if len(partes) > 0:
                    partes[0] = str(nuevo_valor)  # Cambia el primer valor
                lineas_modificadas.append(" ".join(partes) + "\n")
            
            # Sobrescribir el archivo con las líneas modificadas
            with open(ruta_archivo, 'w') as file:
                file.writelines(lineas_modificadas)

# Uso del script
ruta_carpeta = 'C:\\Users\\OPEN SERVICE EIRL\\Documents\\UPN\\Ciclo 7\\Machine learning\\Proyecto_Final_Machine Learning\\Proyecto Final_Machine learning\\dataset\\train\\limon\\abels'# Reemplaza con la ruta de tu carpeta
nuevo_valor = 0  # El valor que quieres poner al inicio de cada línea
cambiar_valor_en_txt(ruta_carpeta, nuevo_valor)
