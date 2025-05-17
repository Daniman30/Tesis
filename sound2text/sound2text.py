import os
import time
import shutil
import ffmpeg
import whisper
import librosa
import numpy as np
from moviepy import *
import subprocess as sp
from pathlib import Path
import tensorflow_hub as hub
from pyannote.audio import Pipeline
from CONSTANTS import RUTA_AUDIO, RUTA_VIDEO, RUTA_TEXTO, HUGGINGFACE_TOKEN

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desactiva GPU
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Evita asignación dinámica de RAM

# region HandleAudio


def extract_audio(name: str):
    """
    Extrae el audio de un archivo de video en formato MKV y lo guarda como archivo WAV.

    Parámetros:
    -----------
    name : str
        Nombre base del archivo de video (sin extensión) ubicado en la ruta global `RUTA_VIDEO`.

    Retorna:
    --------
    str
        Ruta completa del archivo de audio WAV generado.

    Detalles:
    ---------
    - Utiliza ffmpeg para extraer el audio con 2 canales (estéreo) y una frecuencia de muestreo de 44100 Hz.
    - Sobrescribe el archivo de salida si ya existe.
    - Las rutas `RUTA_VIDEO` y `RUTA_AUDIO` deben estar definidas en el entorno global.
    """
    input_mkv = f"{RUTA_VIDEO}/{name}.mkv"
    output_wav = f"{RUTA_AUDIO}/{name}.wav"

    ffmpeg.input(input_mkv).output(output_wav, ac=2,
                                   ar=44100).run(overwrite_output=True)
    return output_wav


def divide_audio(ruta_archivo: str, duracion_segmento: int):
    """
    Divide un archivo de audio en segmentos de duración específica usando ffmpeg.

    Parámetros:
    -----------
    ruta_archivo : str
        Ruta completa del archivo de audio que se desea dividir.
    duracion_segmento : int
        Duración en segundos de cada segmento en que se dividirá el audio.

    Funcionalidad:
    ---------------
    - Verifica que el archivo de audio exista.
    - Crea una carpeta de salida con el nombre base del archivo, dentro de su mismo directorio.
    - Usa ffmpeg para dividir el audio en segmentos sin re-encodear (copia directa).
    - Los segmentos se guardan con formato MKV y nombre secuencial en la carpeta de salida.
    - Maneja errores en la ejecución de ffmpeg mostrando mensajes en consola.

    Ejemplo de uso:
    ---------------
    divide_audio("ruta/a/audio.wav", 120)  # Divide en segmentos de 2 minutos

    """
    if not os.path.isfile(ruta_archivo):
        print("Archivo no encontrado:", ruta_archivo)
        return

    # Obtener nombre base del archivo sin extensión
    nombre_base = os.path.splitext(os.path.basename(ruta_archivo))[0]
    carpeta_salida = os.path.join(os.path.dirname(ruta_archivo), nombre_base)

    # Crear carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Comando ffmpeg para dividir el audio en partes de 2 minutos (120 segundos)
    # -f segment divide el archivo
    # -segment_time 120 define duración por segmento
    # -c copy evita re-encoding
    salida_segmentos = os.path.join(carpeta_salida, f"{nombre_base}_%03d.mkv")
    comando = [
        'ffmpeg',
        '-i', ruta_archivo,
        '-f', 'segment',
        '-segment_time', str(duracion_segmento),
        '-c', 'copy',
        salida_segmentos
    ]

    # Ejecutar el comando
    try:
        sp.run(comando, check=True)
        print("División completada. Archivos guardados en:", carpeta_salida)
    except sp.CalledProcessError as e:
        print("Error al dividir el archivo:", e)


def separate_music_voice(audio_file):
    """
    Separa las pistas de voz y música de un archivo de audio usando Demucs.

    Parámetros:
    -----------
    audio_file : str
        Ruta al archivo de audio que se desea procesar.

    Funcionalidad:
    ---------------
    - Crea una carpeta de salida basada en el nombre del archivo, dentro del mismo directorio.
    - Ejecuta Demucs para separar la voz (vocals) del resto de la música (no_vocals).
    - Mueve los archivos separados (vocals.wav y no_vocals.wav) a la carpeta principal de salida.
    - Elimina la carpeta temporal creada por Demucs.
    
    Retorna:
    --------
    tuple(Path, Path)
        Rutas absolutas a los archivos resultantes vocals.wav y no_vocals.wav.

    Ejemplo de uso:
    ---------------
    vocals_path, music_path = separate_music_voice("audio/song.mp3")

    """
    # Obtener nombre base del archivo sin extensión
    nombre_base = os.path.splitext(os.path.basename(audio_file))[0]
    carpeta_salida = os.path.join(os.path.dirname(audio_file), nombre_base)

    # Crear carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    command = [
        "demucs",
        "--two-stems", "vocals",  # solo separa voz vs resto
        "-o", str(carpeta_salida),
        audio_file
    ]
    sp.run(command)

    # Ruta donde demucs guarda los archivos
    stem_folder = Path(carpeta_salida) / "htdemucs" / nombre_base
    vocals = stem_folder / "vocals.wav"
    no_vocals = stem_folder / "no_vocals.wav"

    # Nuevas rutas deseadas
    final_vocals = Path(carpeta_salida) / "vocals.wav"
    final_no_vocals = Path(carpeta_salida) / "no_vocals.wav"

    # Mover archivos a la carpeta deseada
    shutil.move(str(vocals), str(final_vocals))
    shutil.move(str(no_vocals), str(final_no_vocals))

    # Borrar carpeta intermedia
    shutil.rmtree(Path(carpeta_salida) / "htdemucs")

    return final_vocals, final_no_vocals


def count_files_by_format(ruta, extension):
    """
    Cuenta la cantidad de archivos con una extensión específica en un directorio dado.

    Parámetros:
    -----------
    ruta : str
        Ruta al directorio donde se realizará la búsqueda.
    extension : str
        Extensión de los archivos a contar (puede incluir o no el punto inicial, ej. 'txt' o '.txt').

    Retorna:
    --------
    int
        Número de archivos que tienen la extensión especificada en el directorio dado.

    Ejemplo de uso:
    ---------------
    num_txt = count_files_by_format('/home/user/docs', 'txt')
    """

    if not extension.startswith('.'):
        extension = '.' + extension

    archivos = [
        archivo for archivo in os.listdir(ruta)
        if archivo.endswith(extension) and os.path.isfile(os.path.join(ruta, archivo))
    ]
    return len(archivos)


def process_mkv(name, segment_duration, split=True):
    """
    Procesa un archivo .mkv extrayendo su audio y separando voz de música/efectos.

    Parámetros:
    -----------
    name : str
        Nombre base del archivo .mkv (sin la extensión), ubicado en la ruta definida por RUTA_VIDEO.
    segment_duration : int
        Duración en segundos para dividir el audio si `split` es True.
    split : bool, opcional
        Si es True, divide el audio en segmentos antes de aplicar la separación. 
        Si es False, separa la pista completa sin dividir.

    Funcionalidad:
    --------------
    - Extrae el audio del archivo .mkv y lo guarda como .wav.
    - Si `split` es False:
        - Separa la voz del resto del audio directamente.
        - Muestra las rutas de salida de las pistas separadas.
    - Si `split` es True:
        - Divide el audio en segmentos de duración `segment_duration`.
        - Aplica separación de voz/música con Demucs a cada segmento.
        - Imprime la cantidad de archivos procesados y las rutas finales.

    Retorna:
    --------
    None

    Ejemplo de uso:
    ---------------
    process_mkv("Zootopia", 120, split=True)
    """
    print("🎵 Extrayendo audio del .mkv...")
    audio_path = extract_audio(name)
    # audio_path = f"{RUTA_AUDIO}/{name}.wav"

    if not split:
        vocals, no_vocals = separate_music_voice(audio_path)
        print(f"✅ Voz humana: {vocals}")
        print(f"✅ Resto del audio: {no_vocals}")
        return

    divide_audio(audio_path, segment_duration)

    cantidad = count_files_by_format(f"{RUTA_AUDIO}/{name}", "mkv")
    print(f"Hay {cantidad} archivos .mkv en la ruta {RUTA_AUDIO}/{name}")

    print("🎛️ Separando voz de música y efectos con Demucs...")
    vocals = 0
    no_vocals = 0
    for i in range(cantidad):
        num = f"{i:03}"
        input_wav = f"{RUTA_AUDIO}/{name}/{name}_{num}.mkv"
        vocals, no_vocals = separate_music_voice(input_wav)

    print(f"✅ Voz humana: {vocals}")
    print(f"✅ Resto del audio: {no_vocals}")

# endregion

# region TranscribeVoice


def transcribe_and_diarize(audio_path, model="medium"):
    """
    Realiza la transcripción y diarización de un archivo de audio, 
    asignando fragmentos de texto a los respectivos hablantes.

    Parámetros:
    -----------
    audio_path : str
        Ruta al archivo de audio que se desea procesar (formato compatible como .wav o .mkv).
    model : str, opcional
        Modelo de Whisper a utilizar para la transcripción. 
        Puede ser "base", "small", "medium" o "large". Por defecto es "medium".

    Funcionalidad:
    --------------
    1. Carga un modelo de transcripción de Whisper para convertir audio a texto.
    2. Usa el pipeline de diarización de pyannote-audio para identificar los segmentos de voz y los hablantes.
    3. Asocia cada segmento transcrito con el hablante correspondiente en base a los intervalos temporales.
    
    Retorna:
    --------
    result : list of dict
        Lista de segmentos, cada uno con información sobre:
        - El texto transcrito,
        - El tiempo de inicio y fin,
        - El identificador del hablante.

    Ejemplo de uso:
    ---------------
    result = transcribe_and_diarize("audio/entrevista.wav", model="large")
    """
    # Cargar modelo Whisper
    # "base", "small", "medium", "large"
    whisper_model = whisper.load_model(model)

    # Cargar modelo de diarización
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        # use_auth_token="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"
        use_auth_token=HUGGINGFACE_TOKEN
    )

    # Paso 1: Diarización (detecta quién habla y cuándo)
    diarization = diarization_pipeline(audio_path)

    speaker_segments = []
    for turn in diarization.itertracks(yield_label=True):
        segment = {
            'start': turn[0].start,
            'end': turn[0].end,
            'speaker': turn[2]
        }
        speaker_segments.append(segment)

    # Paso 2: Transcripción completa con Whisper
    whisper_result = whisper_model.transcribe(audio_path, verbose=False)
    segments = whisper_result['segments']

    # Paso 3: Fusionar transcripción con los hablantes
    result = assign_speakers_to_whisper(segments, speaker_segments)
    return result


def assign_speakers_to_whisper(whisper_segments, diarization_segments):
    """
    Asigna a cada segmento transcrito por Whisper el hablante correspondiente 
    utilizando los datos de diarización.

    Parámetros:
    -----------
    whisper_segments : list of dict
        Lista de segmentos generados por Whisper, cada uno con claves como 
        'start', 'end' y 'text', indicando los tiempos de inicio y fin y el texto transcrito.
    
    diarization_segments : list of dict
        Lista de segmentos con información de diarización, donde cada uno contiene 
        'start', 'end' y 'speaker', indicando los intervalos de habla y el hablante asignado.

    Funcionalidad:
    --------------
    - Para cada segmento de Whisper, busca los segmentos de diarización que se solapan en tiempo.
    - Calcula la duración de cada solapamiento y elige al hablante con mayor tiempo compartido.
    - Si no hay solapamientos, asigna "UNKNOWN" como hablante.
    
    Retorna:
    --------
    result : list of dict
        Lista de segmentos transcritos con información adicional del hablante asignado.
        Cada entrada incluye: 'speaker', 'start', 'end' y 'text'.

    Ejemplo:
    --------
    result = assign_speakers_to_whisper(segments_whisper, segments_diarization)
    """
    result = []
    for ws in whisper_segments:
        candidates = []
        for ds in diarization_segments:
            # Si hay solapamiento temporal
            if ds['start'] < ws['end'] and ds['end'] > ws['start']:
                # calculamos cuánto tiempo se solapan
                overlap_start = max(ws['start'], ds['start'])
                overlap_end = min(ws['end'], ds['end'])
                duration = overlap_end - overlap_start
                candidates.append((ds['speaker'], duration))

        # Elegimos el speaker con más tiempo solapado
        if candidates:
            speaker = max(candidates, key=lambda x: x[1])[0]
        else:
            speaker = "UNKNOWN"

        result.append({
            'speaker': speaker,
            'start': ws['start'],
            'end': ws['end'],
            'text': ws['text']
        })

    return result


def save_transcription(transcripcion, ruta_archivo):
    """
    Guarda una transcripción con marcas de tiempo y hablantes en un archivo de texto.

    Parámetros:
    -----------
    transcripcion : list of dict
        Lista de segmentos transcritos, donde cada entrada debe contener las claves:
        - 'start': tiempo de inicio del segmento (en segundos)
        - 'end': tiempo de fin del segmento (en segundos)
        - 'speaker': identificador del hablante (por ejemplo, "SPEAKER_00")
        - 'text': texto transcrito
    
    ruta_archivo : str
        Ruta donde se guardará el archivo de texto. Si los directorios intermedios no existen, se crean.

    Funcionalidad:
    --------------
    - Asegura que el directorio de salida exista.
    - Escribe cada segmento en una línea con el siguiente formato:
        [inicio - fin] hablante: texto

    Ejemplo de salida:
    ------------------
    [3.25 - 5.10] SPEAKER_01: Hola, ¿cómo estás?
    """
    # Convertir a Path object (más robusto para manejo de rutas)
    path_archivo = Path(ruta_archivo)

    # Obtener solo el directorio padre
    directorio = path_archivo.parent

    # Crear el directorio (con parents=True para crear toda la jerarquía si hace falta)
    directorio.mkdir(parents=True, exist_ok=True)

    with open(ruta_archivo, "w", encoding="utf-8") as f:
        for entrada in transcripcion:
            inicio = entrada["start"]
            fin = entrada["end"]
            hablante = entrada["speaker"]
            texto = entrada["text"]
            f.write(f"[{inicio:.2f} - {fin:.2f}] {hablante}: {texto}\n")


def transcribe_voice(name, split=True):
    """
    Transcribe y diariza la voz humana extraída de un archivo de audio, ya sea completo o segmentado,
    y guarda los resultados en archivos de texto.

    Parámetros:
    -----------
    name : str
        Nombre base del archivo o conjunto de archivos de audio a procesar. Se asume que los archivos
        de voz (vocals.wav) se encuentran en rutas relativas a este nombre dentro de `RUTA_AUDIO`.

    split : bool, opcional (por defecto=True)
        Si es True, procesa múltiples segmentos de audio (divididos previamente).
        Si es False, procesa un único archivo de audio completo.

    Funcionalidad:
    --------------
    - Para cada archivo de audio con solo la voz humana (vocals.wav), ejecuta el pipeline de transcripción 
      y diarización.
    - Guarda cada transcripción en un archivo `dialogue_transcription.txt` dentro de la carpeta correspondiente 
      en `RUTA_TEXTO`.

    Estructura esperada:
    --------------------
    Si split=True:
        RUTA_AUDIO/{name}/{name}_000/vocals.wav
        RUTA_AUDIO/{name}/{name}_001/vocals.wav
        ...
    Si split=False:
        RUTA_AUDIO/{name}/vocals.wav

    Salida:
    -------
    Archivos de texto con la transcripción y asignación de hablantes guardados en:
        RUTA_TEXTO/{name}/{name}_XXX/dialogue_transcription.txt  (modo split)
        o
        RUTA_TEXTO/{name}/dialogue_transcription.txt  (modo no dividido)
    """
    if split:
        count = count_files_by_format(f"{RUTA_AUDIO}/{name}", "mkv")
        for i in range(count):
            num = f"{i:03}"
            audio_file = f"{RUTA_AUDIO}/{name}/{name}_{num}/vocals.wav"
            resultado = transcribe_and_diarize(audio_file)

            # Guardar en archivo
            output_txt = f"{RUTA_TEXTO}/{name}/{name}_{num}/dialogue_transcription.txt"
            save_transcription(resultado, output_txt)

        print(f"\n✅ Transcripciones guardadas en: {RUTA_TEXTO}/{name}")
        return

    audio_file = f"{RUTA_AUDIO}/{name}/vocals.wav"
    resultado = transcribe_and_diarize(audio_file)

    # Guardar en archivo
    output_txt = f"{RUTA_TEXTO}/{name}/dialogue_transcription.txt"
    save_transcription(resultado, output_txt)
    print(f"\n✅ Transcripción guardada en: {output_txt}")

# endregion

# region Music2Text


def music2text(name, split=True):
    """
    Detecta y transcribe eventos sonoros no vocales (música, efectos, etc.) en archivos de audio utilizando
    el modelo YAMNet, y exporta los resultados como archivos de subtítulos en formato `.srt`.

    Parámetros:
    -----------
    name : str
        Nombre base del archivo o conjunto de archivos a procesar. Se espera que los archivos de audio
        sin voz humana (no_vocals.wav) estén ubicados en rutas relativas dentro de `RUTA_AUDIO`.

    split : bool, opcional (por defecto=True)
        Si es True, procesa múltiples segmentos de audio (por ejemplo, no_vocals.wav divididos por escena).
        Si es False, procesa un único archivo de audio completo.

    Funcionalidad:
    --------------
    - Carga el modelo YAMNet desde un path local predefinido.
    - Carga la lista de etiquetas de sonidos desde un archivo CSV (class_map.csv).
    - Aplica detección de eventos sonoros (música, ambiente, efectos) usando YAMNet.
    - Exporta los eventos detectados como subtítulos en formato `.srt`.

    Estructura esperada:
    --------------------
    Si split=True:
        RUTA_AUDIO/{name}/{name}_000/no_vocals.wav
        RUTA_AUDIO/{name}/{name}_001/no_vocals.wav
        ...
    Si split=False:
        RUTA_AUDIO/{name}/no_vocals.wav

    Salida:
    -------
    Archivos `.srt` con los eventos musicales/sonoros:
        RUTA_TEXTO/{name}/{name}_XXX/music_caption.srt  (modo split)
        o
        RUTA_TEXTO/{name}/music_caption.srt  (modo no dividido)
    """
    # Cargar modelo YAMNet
    yamnet_model_handle = 'D:/UNI/Tesis/Code/yamnet/'
    yamnet_model = hub.load(yamnet_model_handle)

    # Obtener etiquetas
    class_map_path = 'D:/UNI/Tesis/Code/class_map.csv'
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = f.read().splitlines()
    labels = [row.split(',')[2] for row in class_map[1:]]

    if split:
        count = count_files_by_format(f"{RUTA_AUDIO}/{name}", "mkv")
        for i in range(count):
            num = f"{i:03}"
            audio_path = f"{RUTA_AUDIO}/{name}/{name}_{num}/no_vocals.wav"
            eventos = get_sound_events(
                audio_path, yamnet_model, labels, threshold=0.2)

            # Guardar en archivo
            output_file = f"{RUTA_TEXTO}/{name}/{name}_{num}/music_caption.srt"
            export_to_srt(eventos, output_file)

        print(
            f"✅ Archivos SRT generados como 'music_caption.srt' en {RUTA_TEXTO}/{name}.")
        return

    # Procesamiento
    audio_path = f"{RUTA_AUDIO}/{name}/no_vocals.wav"
    eventos = get_sound_events(audio_path, yamnet_model, labels, threshold=0.2)

    # Exportar como subtítulos .srt
    output_file = f"{RUTA_TEXTO}/{name}/music_caption.srt"
    export_to_srt(eventos, output_file)

    print(f"✅ Archivo SRT generado en {output_file}.")


def load_audio(file_path):
    """
    Carga un archivo de audio y lo convierte en una señal monoaural con una frecuencia de muestreo de 16 kHz.

    Parámetros:
    -----------
    file_path : str
        Ruta al archivo de audio que se desea cargar.

    Retorna:
    --------
    waveform : np.ndarray
        Array unidimensional con la señal de audio normalizada.
    
    Notas:
    ------
    - Utiliza la función `librosa.load` con los parámetros `sr=16000` y `mono=True` para garantizar
      una frecuencia de muestreo consistente y una única pista de audio.
    - Es adecuado para tareas de análisis de audio como transcripción, clasificación o extracción de características.
    """
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    return waveform


def get_sound_events(audio_path, yamnet_model, labels, threshold=0.2):
    """
    Detecta eventos sonoros en un archivo de audio utilizando el modelo YAMNet.

    Esta función carga el audio desde la ruta especificada, lo procesa con YAMNet y extrae 
    los eventos sonoros cuya probabilidad supera un umbral dado.

    Parámetros:
    -----------
    audio_path : str
        Ruta al archivo de audio (.wav) que se desea analizar.
    
    yamnet_model : tensorflow_hub.KerasLayer
        Modelo YAMNet cargado desde TensorFlow Hub.
    
    labels : list of str
        Lista de etiquetas asociadas a las clases que predice YAMNet.
    
    threshold : float, opcional
        Umbral mínimo de probabilidad para considerar una predicción válida. 
        Valor por defecto: 0.2

    Retorna:
    --------
    predictions : list of tuples
        Lista de eventos detectados en el formato (timestamp, etiqueta, puntuación),
        donde:
            - timestamp (float) es el tiempo en segundos dentro del audio,
            - etiqueta (str) es el nombre del evento sonoro detectado,
            - puntuación (float) es la confianza de la predicción.

    Notas:
    ------
    - Cada frame del modelo YAMNet representa aproximadamente 0.96 segundos de audio.
    - El audio se remuestrea internamente a 16 kHz y se convierte en mono si no lo está.
    - Esta función imprime información útil sobre el número de frames y la duración total del audio.
    """
    waveform = load_audio(audio_path)

    scores, embeddings, spectrogram = yamnet_model(waveform)
    print(f"[INFO] Número de frames que da el modelo: {scores.shape[0]}")
    print(
        f"[INFO] Tiempo total estimado por YAMNet: {scores.shape[0] * 0.96:.2f} segundos")

    duration = len(waveform) / 16000
    print(f"[INFO] Duración del audio: {duration:.2f} segundos")
    scores_np = scores.numpy()
    num_frames = scores_np.shape[0]
    time_per_frame = duration / num_frames

    predictions = []
    for i, score in enumerate(scores_np):
        top_index = np.argmax(score)
        top_score = score[top_index]
        if top_score > threshold:
            time_stamp = i * time_per_frame  # en proporción al audio real
            predictions.append((time_stamp, labels[top_index], top_score))
    return predictions


def seconds_to_srt_time(seconds):
    """
    Convierte un valor de tiempo en segundos a formato de tiempo SRT.

    El formato SRT es utilizado en archivos de subtítulos y sigue el formato:
    `HH:MM:SS,mmm`, donde:
        - HH: horas con dos dígitos
        - MM: minutos con dos dígitos
        - SS: segundos con dos dígitos
        - mmm: milisegundos con tres dígitos

    Parámetros:
    -----------
    seconds : float
        Tiempo en segundos que se desea convertir.

    Retorna:
    --------
    str
        Cadena con el tiempo en formato SRT.

    Ejemplo:
    --------
    >>> seconds_to_srt_time(75.345)
    '00:01:15,345'
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def export_to_srt(predictions, output_file, duration=1.5):
    """
    Exporta eventos sonoros a un archivo de subtítulos en formato SRT.

    Cada evento se representa como un subtítulo con una duración fija y una etiqueta de sonido 
    entre corchetes. Esta función es útil para generar subtítulos automáticos de sonidos no verbales 
    como música, efectos o ruidos ambientales.

    Parámetros:
    -----------
    predictions : list of tuples
        Lista de eventos sonoros en el formato (timestamp, etiqueta, confianza), 
        donde:
            - timestamp (float): tiempo de inicio del evento en segundos.
            - etiqueta (str): nombre del evento detectado.
            - confianza (float): valor de probabilidad asociado (no utilizado en el archivo final).
    
    output_file : str
        Ruta al archivo `.srt` donde se guardarán los subtítulos generados.
    
    duration : float, opcional
        Duración de cada subtítulo en segundos. Por defecto es 1.5 segundos.

    Formato de salida:
    ------------------
    El archivo SRT generado tendrá entradas numeradas con los tiempos de inicio y fin en formato SRT,
    y una línea con la etiqueta del evento, por ejemplo:

        1
        00:00:01,500 --> 00:00:03,000
        [Dog barking]

    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (time, label, conf) in enumerate(predictions, start=1):
            start = seconds_to_srt_time(time)
            end = seconds_to_srt_time(time + duration)
            f.write(f"{idx}\n{start} --> {end}\n[{label}]\n\n")

# endregion


def sound2text(name, split=True, segment_duration=120):
    """
    Orquesta el proceso completo de extracción, segmentación, transcripción
    y conversión de audio a texto a partir de un archivo MKV.

    Este proceso incluye:
    1. Extracción y segmentación del audio del archivo MKV.
    2. Transcripción de la voz humana detectada en el audio.
    3. Conversión de la música y efectos de sonido a texto (subtítulos).

    Parámetros:
    -----------
    name : str
        Nombre base del archivo MKV (sin extensión) y carpeta relacionada.
    split : bool, opcional
        Indica si se debe dividir el audio en segmentos para procesamiento
        (por defecto es True).
    segment_duration : int, opcional
        Duración en segundos de cada segmento si se divide el audio
        (por defecto es 120 segundos).

    Retorna:
    --------
    None

    Imprime en consola el progreso y el tiempo total de ejecución del proceso.
    """

    # Registrar el tiempo de inicio
    inicio = time.time()

    # """Función principal que orquesta el proceso completo."""
    print(f"\nProcessing MKV...\n")
    process_mkv(name, segment_duration, split)

    print(f"\nTranscribing voice...\n")
    transcribe_voice(name, split)

    print(f"\nConverting music to text...\n")
    music2text(name, split)

    # Registrar el tiempo de finalización
    fin = time.time()

    # Calcular e imprimir el tiempo transcurrido
    tiempo_transcurrido = fin - inicio
    print(f"El código tardó {tiempo_transcurrido} segundos en ejecutarse")

# if __name__ == '__main__':
#     name = "Zootopia"
#     sound2text(name)
