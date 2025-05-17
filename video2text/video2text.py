import os
import cv2
import csv
import torch
import requests
from PIL import Image
from pathlib import Path
from CONSTANTS import RUTA_VIDEO, RUTA_IMAGE, RUTA_TEXTO
from huggingface_hub import snapshot_download
from sound2text.sound2text import count_files_by_format
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration


def opencv(video_name, frames_por_segundo=1):
    """
    Extrae y guarda un frame por segundo de un video usando OpenCV.

    Parámetros:
    ----------
    video_name : str
        Nombre del archivo de video (incluyendo la extensión) ubicado en la ruta definida por RUTA_VIDEO.

    frames_por_segundo : int, opcional (por defecto = 1)
        Cantidad de frames que se van a extraer por cada segundo del video.

    Funcionalidad:
    -------------
    - Carga el video desde la ruta especificada.
    - Verifica que el archivo se pueda abrir correctamente.
    - Obtiene los fotogramas por segundo (FPS) del video.
    - Extrae un frame por segundo (puede ajustarse cambiando 'frames_por_segundo').
    - Guarda cada frame extraído como una imagen `.jpg` en el directorio definido por RUTA_IMAGE.
    - Crea el directorio de salida si no existe.

    Salida:
    ------
    - Muestra en consola los FPS del video.
    - Imprime mensajes por cada frame guardado o si ocurre un error.
    - Al finalizar, informa cuántos frames se procesaron y cuántos se guardaron.
    """
    video_path = os.path.join(RUTA_VIDEO, video_name)

    # Asegurar que el directorio de salida existe
    output_dir = RUTA_IMAGE
    os.makedirs(output_dir, exist_ok=True)  # Crea la carpeta si no existe

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error al abrir el video. Verifica la ruta:", video_path)
        exit()

    # Obtener los FPS del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS del video: {fps}")

    # Calcular el intervalo de frames a capturar
    intervalo = int(round(fps / frames_por_segundo)
                    ) if frames_por_segundo > 0 else 1

    frame_count = 0
    frames_extraidos = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Solo guardar el frame si es el que corresponde según el intervalo
        if frame_count % intervalo == 0:
            output_path = os.path.join(
                output_dir, f"frame_{frames_extraidos:04d}.jpg")
            if not cv2.imwrite(output_path, frame):
                print(
                    f"❌ Error al guardar {output_path}. ¿Permisos? ¿Ruta válida?")
            else:
                print(f"✅ Guardado: {output_path}")
                frames_extraidos += 1

        frame_count += 1

    cap.release()
    print(f"De {frame_count} frames fueron extraídos {frames_extraidos} frames y guradados en {output_dir}.")


def image_description(image_path, output_path):
    """
    Genera descripciones automáticas para imágenes utilizando el modelo BLIP y actualiza un archivo CSV con dicha información.

    Parámetros:
    -----------
    image_path : str
        Ruta al directorio que contiene las imágenes (formato .jpg) a procesar.

    output_path : str
        Ruta al directorio donde se encuentra el archivo `scene_frames.csv` con los nombres de los frames y timestamps,
        y donde se guardarán los resultados (descripciones generadas).

    Funcionalidad:
    --------------
    - Verifica si el modelo BLIP ya está descargado localmente; si no, lo descarga automáticamente.
    - Carga el modelo y su procesador en GPU o CPU, según disponibilidad.
    - Lee un archivo `scene_frames.csv` que debe contener al menos dos columnas: `frame_name` y `timestamp`.
    - Para cada imagen listada:
        - Verifica que el archivo exista.
        - Procesa la imagen y genera una descripción textual usando el modelo.
        - Agrega la descripción como una tercera columna.
    - Guarda el CSV actualizado con la columna adicional `description`.

    Resultados:
    -----------
    - Un nuevo archivo CSV (`scene_frames.csv`) con las descripciones generadas para cada imagen.
    - Mensajes informativos impresos en consola sobre el progreso del procesamiento.

    Notas:
    ------
    - El modelo utilizado es `Salesforce/blip-image-captioning-base`.
    - La función depende de la existencia de los archivos e imágenes especificados.
    - Requiere conexión a internet para la descarga inicial del modelo (si no está presente).
    """
    # Ruta local donde guardar el modelo
    MODEL_DIR = Path("./blip-base")

    # Verificar si el modelo ya está descargado
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print("🔽 Modelo no encontrado localmente. Descargando...")
        snapshot_download(
            repo_id="Salesforce/blip-image-captioning-base",
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False  # importante para compatibilidad en algunos sistemas
        )
        print("✅ Modelo descargado y guardado en:", MODEL_DIR)
    else:
        print("📦 Modelo ya disponible localmente en:", MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Modelo cargado en: {device}")

    # Cargar modelo
    print("📦 Cargando procesador...")
    processor = BlipProcessor.from_pretrained(str(MODEL_DIR))
    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

    print("🧠 Cargando modelo...")
    model = BlipForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    # model = Blip2ForConditionalGeneration.from_pretrained(
    #     "Salesforce/blip2-flan-t5-xl",
    #     load_in_4bit=True,
    #     device_map="auto"
    # ).to(device)

    # Archivos donde se guardarán las descripciones
    output_txt = os.path.join(output_path, "descripciones.txt")
    output_csv = os.path.join(output_path, "scene_frames.csv")

    # Contar imágenes
    num_images = count_files_by_format(image_path, 'jpg')
    print(f"📷 Número de imágenes encontradas: {num_images}")

    # Leer los datos existentes del CSV
    with open(output_csv, "r", encoding="utf-8") as f_csv:
        reader = csv.reader(f_csv)
        header = next(reader)
        rows = list(reader)

    updated_rows = []

    # Procesar cada fila del CSV
    for row in rows:
        frame_name, timestamp = row[0], row[1]
        image_file = os.path.join(image_path, frame_name)

        if not os.path.exists(image_file):
            print(f"⚠️ Archivo no encontrado: {image_file}")
            updated_rows.append([frame_name, timestamp, ""])
            continue

        print(f"🖼️ Procesando imagen: {image_file}")
        image = Image.open(image_file).convert("RGB")

        # Procesar imagen
        inputs = processor(image, return_tensors="pt").to(model.device)

        # Generar descripción
        print("⚙️ Generando descripción...")
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        print(f"📝 Descripción generada: {caption}")
        updated_rows.append([frame_name, timestamp, caption])

    # Escribir nuevo CSV con la columna de descripción
    with open(output_csv, "w", newline='', encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame_name", "timestamp", "description"])
        writer.writerows(updated_rows)

    print(f"💾 Archivo CSV actualizado con descripciones: {output_csv}")


def describe_image_with_huggingface_api(image_path, hf_token):
    """
    Genera una descripción automática de una imagen utilizando la API de Hugging Face con el modelo BLIP.

    Parámetros:
    -----------
    image_path : str
        Ruta al archivo de imagen (preferiblemente en formato JPG o PNG) que se desea describir.

    hf_token : str
        Token de autenticación personal de Hugging Face con permisos para acceder a la API de inferencia.

    Funcionalidad:
    --------------
    - Lee la imagen especificada como un archivo binario.
    - Envía la imagen al modelo `Salesforce/blip-image-captioning-base` alojado en Hugging Face mediante una petición POST.
    - Procesa la respuesta y extrae el texto generado si está disponible.

    Retorna:
    --------
    str or None:
        - Cadena con la descripción generada de la imagen si la solicitud fue exitosa y la respuesta fue válida.
        - None si hubo un error en la solicitud o si la respuesta no contiene el texto esperado.

    Notas:
    ------
    - Es necesario tener conexión a internet para acceder a la API.
    - La función imprime mensajes en consola para informar sobre el progreso y posibles errores.
    - El token (`hf_token`) debe tener permisos para usar el modelo `Salesforce/blip-image-captioning-base`.

    Ejemplo de uso:
    ---------------
    >>> descripcion = describe_image_with_huggingface_api("frame_0001.jpg", "hf_xxxxxxxxxxxxxxxxxx")
    >>> print(descripcion)
    "A dog sitting on a couch in a living room."
    """
    # Endpoint del modelo de captioning de imágenes
    api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

    # Leer imagen en binario
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    print("⏳ Enviando imagen al modelo BLIP-2 vía API...")
    response = requests.post(api_url, headers=headers, data=image_bytes)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            description = result[0]["generated_text"]
            print("📝 Descripción generada:", description)
            return description
        else:
            print("⚠️ Respuesta inesperada:", result)
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

    return None


def video2text(video_name):
    """
    Genera descripciones textuales de las escenas de un video a partir de sus imágenes.

    Parámetros:
    -----------
    video_name : str
        Nombre del archivo de video (sin la ruta) del cual se desea generar descripciones.

    Funcionalidad:
    --------------
    - Define la ruta donde se encuentran las imágenes del video segmentado por escenas (`RUTA_IMAGE/{video_name}_by_scene`).
    - Define la ruta de salida donde se guardarán las descripciones generadas (`RUTA_TEXTO/{video_name}`).
    - Llama a la función `image_description` para generar descripciones de cada imagen de escena utilizando el modelo BLIP de forma local.
    - (Opcional) Se puede cambiar el método de descripción usando la API de Hugging Face comentando/descomentando líneas relevantes.

    Notas:
    ------
    - Se espera que las imágenes del video ya hayan sido extraídas previamente y estén organizadas en carpetas por escena.
    - Las rutas `RUTA_IMAGE` y `RUTA_TEXTO` deben estar definidas globalmente en el entorno.
    - El modelo BLIP debe estar disponible localmente o será descargado al momento de la ejecución de `image_description`.

    Ejemplo de uso:
    ---------------
    >>> video2text("Zootopia")
    # Esto generará un archivo `descripciones.txt` y un CSV con las descripciones para cada escena del video.
    """
    # opencv(video_name)
    path = f"{RUTA_IMAGE}/{video_name}_by_scene"
    output_path = f"{RUTA_TEXTO}/{video_name}"
    image_description(path, output_path)
    # describe_image_with_huggingface_api(path, HUGGINGFACE_TOKEN)

# if __name__ == '__main__':
#     start = time.time()

#     name = "Zootopia"
#     video2text(name)

#     end = time.time()
#     total = end - start
#     print(f"Se demoro {total} segundos")
