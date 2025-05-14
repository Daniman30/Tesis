import os
import cv2
import csv
import time
import tqdm
import torch
import requests
from PIL import Image
from pathlib import Path
from CONSTANTS import RUTA_VIDEO, RUTA_IMAGE, RUTA_TEXTO
from huggingface_hub import snapshot_download
from sound2text.sound2text import count_files_by_format
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration


def opencv(video_name):
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

    # Cuántos frames quieres por segundo (ajusta este valor)
    frames_por_segundo = 1

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
