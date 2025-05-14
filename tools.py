import os


def elegir_mkv(ruta):
    # Obtener todos los archivos .mkv
    archivos_mkv = [f[:-4] for f in os.listdir(ruta) if f.endswith('.mkv')]

    if not archivos_mkv:
        print("No se encontraron archivos .mkv en la ruta especificada.")
        return None

    # Mostrar lista numerada
    print("Archivos .mkv encontrados:")
    for i, nombre in enumerate(archivos_mkv):
        print(f"{i + 1}. {nombre}")
    print("0. Para salir")

    # Solicitar selección
    while True:
        try:
            opcion = int(input("Elige un número: "))
            if 1 <= opcion <= len(archivos_mkv):
                name = archivos_mkv[opcion - 1]
                print(f"Seleccionaste: {name}")
                return name
            elif opcion == 0:
                return ''
            else:
                print("Número fuera de rango. Intenta de nuevo.")
        except ValueError:
            print("Entrada inválida. Ingresa un número.")


def segundos_a_hora(segundos):
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segundos_restantes = segundos % 60
    return f"{horas:02}:{minutos:02}:{segundos_restantes:02}"
