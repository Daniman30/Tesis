import os
import time
import ffmpeg
import whisper
import shutil
import librosa
import requests
import numpy as np
import soundfile as sf
import tensorflow as tf
import subprocess as sp
import tensorflow_hub as hub
from moviepy import *
from pathlib import Path
from pyannote.audio import Pipeline
from spleeter.separator import Separator
from CONSTANTS import RUTA_AUDIO, RUTA_VIDEO, RUTA_TEXTO, HUGGINGFACE_TOKEN

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desactiva GPU
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Evita asignación dinámica de RAM

# region HandleAudio


def extract_audio(name: str):
    """
    Extrae el audio de un archivo de video en formato mkv y lo guarda en formato wav 

    Args:
        name (str): Nombre del archivo de video en la RUTA_VIDEO (.mkv)
    """
    input_mkv = f"{RUTA_VIDEO}/{name}.mkv"
    output_wav = f"{RUTA_AUDIO}/{name}.wav"

    ffmpeg.input(input_mkv).output(output_wav, ac=2,
                                   ar=44100).run(overwrite_output=True)
    return output_wav


def divide_audio(ruta_archivo: str, duracion_segmento: int):
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
    Cuenta la cantidad de archivos con una extensión específica en una ruta.

    Parámetros:
        ruta (str): Ruta del directorio donde buscar.
        extension (str): Extensión del archivo, por ejemplo '.wav' o '.mkv'.

    Retorna:
        int: Cantidad de archivos con esa extensión.
    """
    if not extension.startswith('.'):
        extension = '.' + extension

    archivos = [
        archivo for archivo in os.listdir(ruta)
        if archivo.endswith(extension) and os.path.isfile(os.path.join(ruta, archivo))
    ]
    return len(archivos)


def process_mkv(name, segment_duration, split=True):
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
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    return waveform


def get_sound_events(audio_path, yamnet_model, labels, threshold=0.2):
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
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def export_to_srt(predictions, output_file, duration=1.5):
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (time, label, conf) in enumerate(predictions, start=1):
            start = seconds_to_srt_time(time)
            end = seconds_to_srt_time(time + duration)
            f.write(f"{idx}\n{start} --> {end}\n[{label}]\n\n")

# endregion


def sound2text(name, split=True, segment_duration=120):
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
