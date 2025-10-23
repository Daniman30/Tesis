import os
import cv2
import csv
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect import VideoManager, SceneManager
from skimage.metrics import structural_similarity as ssim
from CONSTANTS import RUTA_VIDEO, RUTA_IMAGE, RUTA_TEXTO


def extract_significant_frames(video_path, output_folder, threshold=0.6):
    """
    Extrae y guarda los fotogramas más significativos de un video en función de los cambios de escena.

    Compara histogramas en escala de grises entre frames consecutivos y guarda aquellos cuya diferencia
    supera un umbral especificado, indicando un cambio visual relevante. Los frames se almacenan en una
    carpeta de salida como archivos JPG.

    Parámetros:
    -----------
    video_path : str
        Ruta al archivo de video del que se extraerán los fotogramas.
    
    output_folder : str
        Carpeta donde se guardarán los fotogramas significativos en formato JPG.
        Se crea automáticamente si no existe.
    
    threshold : float, opcional (por defecto = 0.6)
        Umbral de diferencia entre histogramas (Bhattacharyya distance) a partir del cual se considera
        que un frame representa un cambio visual significativo respecto al anterior.

    Funcionalidad:
    --------------
    - Analiza cuadro por cuadro el video indicado.
    - Compara cada frame con el anterior utilizando histogramas en escala de grises.
    - Si la diferencia entre histogramas excede el umbral definido, guarda el frame como imagen.
    - Los frames significativos se almacenan en la carpeta especificada, con nombres secuenciales como
      `frame_0000.jpg`, `frame_0001.jpg`, etc.
    - Al finalizar, se imprime en consola cuántos frames fueron guardados.

    Ejemplo:
    --------
    >>> extract_significant_frames("video.mp4", "./frames_destacados", threshold=0.5)
    Se guardaron 12 frames significativos en './frames_destacados'.
    """
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
    """
    Determina si un fotograma es borroso utilizando la varianza del operador de Laplace.

    Convierte el fotograma a escala de grises y calcula la varianza del Laplaciano, 
    una medida común de enfoque o nitidez. Si la varianza es menor que el umbral 
    especificado, el fotograma se considera borroso.

    Parámetros:
    -----------
    frame : np.ndarray
        Imagen en formato BGR (como se obtiene de OpenCV) que se desea evaluar.
    
    threshold : float, opcional (por defecto = 100.0)
        Umbral de varianza del Laplaciano por debajo del cual se considera que la imagen está borrosa.
        Valores más altos hacen el criterio más estricto.

    Funcionalidad:
    --------------
    - Convierte el fotograma a escala de grises.
    - Calcula la varianza de la imagen filtrada con el operador Laplaciano.
    - Compara la varianza con el umbral proporcionado.
    - Devuelve `True` si la imagen se considera borrosa, `False` en caso contrario.

    Ejemplo:
    --------
    >>> is_blurry(frame)
    False
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold


def extract_better_frames(video_path, output_folder, ssim_threshold=0.8, blur_threshold=100.0):
    """
    Extrae y guarda fotogramas significativos y nítidos de un video, basándose en la similitud estructural 
    (SSIM) entre frames y una medida de desenfoque.

    Esta función compara cada 5 fotogramas con el anterior usando el índice SSIM para detectar cambios 
    significativos. Solo se guardan los fotogramas que sean visualmente distintos (por debajo del umbral 
    de SSIM) y que no estén borrosos (según la varianza del operador Laplaciano).

    Parámetros:
    -----------
    video_path : str
        Ruta al archivo de video del cual se extraerán los fotogramas.
    
    output_folder : str
        Carpeta donde se guardarán los fotogramas seleccionados como imágenes JPG.
    
    ssim_threshold : float, opcional (por defecto = 0.8)
        Umbral del índice de similitud estructural. Si la similitud entre dos frames consecutivos 
        es menor que este valor, se considera que hay un cambio significativo.

    blur_threshold : float, opcional (por defecto = 100.0)
        Umbral de desenfoque. Se descartan los frames cuya nitidez esté por debajo de este valor 
        (varianza del Laplaciano baja).

    Funcionalidad:
    --------------
    - Crea la carpeta de salida si no existe.
    - Lee el video y analiza un frame de cada 5 para mejorar el rendimiento.
    - Compara la similitud del frame actual con el anterior usando SSIM.
    - Verifica que el frame no esté borroso mediante la función `is_blurry`.
    - Guarda el frame como imagen si pasa ambos filtros.

    Ejemplo:
    --------
    >>> extract_better_frames("video.mp4", "frames_output")
    Se guardaron 12 frames significativos y nítidos.
    """
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
    """
    Detecta escenas en un video basándose en cambios de contenido y guarda una imagen representativa 
    por escena, además de generar un archivo CSV con los nombres de los frames y sus timestamps.

    Parámetros:
    -----------
    name : str
        Prefijo para nombrar las imágenes guardadas de cada escena.
    
    video_path : str
        Ruta al archivo de video que se analizará para detectar escenas.
    
    output_folder : str
        Carpeta donde se guardarán las imágenes representativas de cada escena detectada.
    
    csv_folder : str
        Carpeta donde se guardará el archivo CSV que contiene los nombres de los frames y sus timestamps.
    
    threshold : float, opcional (por defecto=30.0)
        Umbral de sensibilidad para detectar cambios de contenido entre escenas. Valores más bajos 
        detectan más cambios (mayor sensibilidad).

    Funcionalidad:
    --------------
    - Inicializa un gestor de video y un detector de escenas basado en cambios de contenido.
    - Detecta las escenas en el video usando el umbral dado.
    - Guarda una imagen representativa por cada escena detectada en la carpeta de salida.
    - Crea un archivo CSV con el nombre de cada frame guardado y su timestamp en formato HH:MM:SS.mmm.
    - Libera los recursos del gestor de video.

    Ejemplo:
    --------
    >>> extract_scenes("video1", "videos/video1.mp4", "output/images", "output/csv")
    12 escenas detectadas.
    """    
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


def one_fps(name, video_path, output_folder, csv_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0  # Contador consecutivo para los nombres de los archivos

    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    # Ruta del archivo CSV
    csv_path = os.path.join(csv_folder, 'scene_frames.csv')

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_name', 'timestamp'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Guardar un frame cada 24 fotogramas
            if count % 24 == 0:
                # Obtener el tiempo actual en milisegundos
                millisec = cap.get(cv2.CAP_PROP_POS_MSEC)
                seconds = millisec / 1000
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                msecs = int(millisec % 1000)
                timestamp_str = f"{hours:02d}:{minutes:02d}:{secs:02d}.{msecs:03d}"

                # Guardar imagen
                frame_filename = os.path.join(output_folder, f"{name}-Scene-{saved_count:04d}-01.jpg")
                cv2.imwrite(frame_filename, frame)

                # Escribir en CSV
                writer.writerow([f"{name}-Scene-{saved_count:04d}-01.jpg", timestamp_str])

                saved_count += 1  # Incrementa solo cuando guardas un frame

            count += 1

    print(f"Total de frames leídos: {count}")
    print(f"Total de frames guardados: {saved_count}")
    cap.release()


def frame_extraction(name):
    """
    Extrae frames significativos o escenas de un video dado, guardando imágenes y metadatos asociados.

    Parámetros:
    -----------
    name : str
        Nombre base del video (sin extensión) que se utilizará para construir rutas de entrada y salida.

    Funcionamiento:
    ---------------
    - Construye las rutas para el video, la carpeta donde se guardarán las imágenes extraídas y la carpeta para los archivos CSV.
    - Utiliza la función `extract_scenes` para detectar y guardar escenas con un umbral de sensibilidad ajustable (por defecto 30.0).
    """
    video_path = f"{RUTA_VIDEO}/{name}.mkv"
    output_folder = f"{RUTA_IMAGE}/{name}_by_scene"
    csv_folder = f"{RUTA_TEXTO}/{name}"
    # extract_significant_frames(video_path, output_folder, threshold=0.6)
    # extract_better_frames(video_path, output_folder, ssim_threshold=0.8, blur_threshold=100)

    # threshold modificable (20 para mas detalle, 40 para menos frames)
    # extract_scenes(name, video_path, output_folder, csv_folder, threshold=30.0)
    one_fps(name, video_path, output_folder, csv_folder)

if __name__ == "main":
    name = "Zootopia"
    frame_extraction(name)
