import os
import cv2
import csv
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect import VideoManager, SceneManager
from scenedetect.frame_timecode import FrameTimecode
from skimage.metrics import structural_similarity as ssim
from CONSTANTS import RUTA_VIDEO, RUTA_IMAGE, RUTA_TEXTO


def extract_significant_frames(video_path, output_folder, threshold=0.6):
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Cargar el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    # Leer el primer frame
    ret, prev_frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        return

    prev_hist = cv2.calcHist([cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)], [
                             0], None, [256], [0, 256])
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Comparar histograma actual con el anterior
        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

        if diff > threshold:
            # Guardar frame si la diferencia es suficiente
            output_path = os.path.join(
                output_folder, f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_idx += 1
            prev_hist = hist  # Actualizar histograma previo

        frame_idx += 1

    cap.release()
    print(
        f"Se guardaron {saved_idx} frames significativos en '{output_folder}'.")


def is_blurry(frame, threshold=100.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold


def extract_better_frames(video_path, output_folder, ssim_threshold=0.8, blur_threshold=100.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("No se pudo leer el video.")
        return

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 5 != 0:  # analizar solo 1 de cada 5 frames para ganar velocidad
            frame_idx += 1
            continue

        similarity = ssim(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if similarity < ssim_threshold and not is_blurry(frame, blur_threshold):
            out_path = os.path.join(
                output_folder, f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
            prev_frame = frame
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Se guardaron {saved_idx} frames significativos y nítidos.")


def extract_scenes(name, video_path, output_folder, csv_folder, threshold=30.0):
    # Crear el gestor de video
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(
        threshold=threshold))  # sensibilidad del cambio

    # Iniciar y analizar
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Obtener las escenas detectadas
    scene_list = scene_manager.get_scene_list()
    print(f"{len(scene_list)} escenas detectadas.")

    # Asegurarse de que exista la carpeta de salida
    os.makedirs(output_folder, exist_ok=True)

    # Guardar una imagen por escena
    save_images(scene_list, video_manager, num_images=1,
                output_dir=output_folder, show_progress=True)

    # Escribir CSV con nombres y timestamps
    csv_path = os.path.join(csv_folder, 'scene_frames.csv')
    fps = video_manager.get_framerate()

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_name', 'timestamp'])

        for i, (start_time, _) in enumerate(scene_list):
            # save_images usa este formato
            frame_name = f'{name}-Scene-{i+1:04d}-01.jpg'
            timestamp = start_time.get_timecode()  # Formato HH:MM:SS.mmm
            writer.writerow([frame_name, timestamp])

    video_manager.release()


def frame_extraction(name):
    video_path = f"{RUTA_VIDEO}/{name}.mkv"
    output_folder = f"{RUTA_IMAGE}/{name}_by_scene"
    csv_folder = f"{RUTA_TEXTO}/{name}"
    # extract_significant_frames(video_path, output_folder, threshold=0.6)
    # extract_better_frames(video_path, output_folder, ssim_threshold=0.8, blur_threshold=100)

    # threshold modificable (20 para mas detalle, 40 para menos frames)
    extract_scenes(name, video_path, output_folder, csv_folder, threshold=30.0)

# if __name__ == "main":
#     name = "Zootopia"
#     frame_extraction(name)
