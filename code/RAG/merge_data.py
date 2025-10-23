import re
import csv
import json
from pathlib import Path
from datetime import timedelta, datetime
from CONSTANTS import RUTA_TEXTO, DURATION_SECONDS


def read_text_file(file_path):
    if Path(file_path).exists():
        return Path(file_path).read_text(encoding="utf-8").strip()
    return ""


def parse_srt(srt_content):
    entries = []
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            times = lines[1].strip()
            start, end = [t.replace(',', '.') for t in times.split(' --> ')]
            text = '\n'.join(lines[2:])
            entries.append({
                "start": start.strip(),
                "end": end.strip(),
                "text": text.strip()
            })
    return entries


def time_to_seconds(time_str):
    h, m, s = map(float, time_str.replace(',', '.').split(':'))
    return h * 3600 + m * 60 + s


def parse_timestamp(timestamp_str):
    """Convierte string flexible de tiempo a datetime."""
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%-H:%M:%S", "%-H:%M:%S.%f"):
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    # Último intento: rellenar ceros y probar con .%f
    if '.' not in timestamp_str:
        timestamp_str += '.000'
    try:
        return datetime.strptime(timestamp_str.zfill(12), "%H:%M:%S.%f")
    except ValueError as e:
        raise ValueError(
            f"No se pudo interpretar el timestamp: {timestamp_str}") from e


def extract_subtitles_in_range(subs, start_time, end_time):
    start_sec = time_to_seconds(start_time)
    end_sec = time_to_seconds(end_time)
    return '\n'.join(
        sub["text"] for sub in subs
        if time_to_seconds(sub["start"]) < end_sec and time_to_seconds(sub["end"]) > start_sec
    )


def guess_timestamps(scene_index):
    start_sec = scene_index * DURATION_SECONDS
    end_sec = start_sec + DURATION_SECONDS
    return str(timedelta(seconds=start_sec)), str(timedelta(seconds=end_sec))


def read_visual_descriptions_from_csv(csv_path):
    descriptions_by_scene = {}
    if Path(csv_path).exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = row["timestamp"].replace(',', '.')
                description = row["description"].strip()
                time_sec = time_to_seconds(timestamp)
                scene_index = int(time_sec // DURATION_SECONDS)
                descriptions_by_scene.setdefault(
                    scene_index, []).append(description)
    return descriptions_by_scene


def extract_visual_descriptions_in_range_from_csv(start_time, end_time, csv_path):
    """
    Extrae descripciones visuales dentro de un rango de tiempo desde un archivo CSV.

    Parámetros:
    - start_time (str): tiempo inicial en formato 'HH:MM:SS.mmm'
    - end_time (str): tiempo final en formato 'HH:MM:SS.mmm'
    - csv_path (str o Path): ruta al archivo CSV

    Devuelve:
    - List[str]: lista de descripciones dentro del rango
    """
    start_dt = parse_timestamp(start_time)
    end_dt = parse_timestamp(end_time)
    descriptions_in_range = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                ts = parse_timestamp(row['timestamp'])
                if start_dt <= ts <= end_dt:
                    descriptions_in_range.append(row['description'].strip())
            except (ValueError, KeyError):
                continue  # Ignora filas con errores de formato

    return descriptions_in_range


def clean_lines(lines):
    return [line.strip() for line in lines if line.strip() and not line.strip().isdigit() and "-->" not in line]


def remove_consecutive_duplicates(lines):
    cleaned = []
    previous = None
    for line in lines:
        if line != previous:
            cleaned.append(line)
            previous = line
    return cleaned


def create_movie_json(path, movie_name):
    output = {
        "movie_name": movie_name,
        "metadata": {},
        "scenes": []
    }

    base_path = Path(path)

    # Metadata
    metadata_path = base_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            output["metadata"] = json.load(f)

    # Subtítulos globales
    srt_path = base_path / f"{movie_name}.srt"
    global_subs = parse_srt(read_text_file(srt_path))

    # Descripciones visuales desde CSV
    csv_path = base_path / "scene_frames.csv"
    visual_descriptions = read_visual_descriptions_from_csv(csv_path)

    # Detectar carpetas por escena
    pattern = re.compile(rf"{re.escape(movie_name)}_(\d{{3}})")
    for item in sorted(base_path.iterdir()):
        match = pattern.fullmatch(item.name)
        if item.is_dir() and match:
            scene_id = int(match.group(1))

            start_time, end_time = guess_timestamps(scene_id)

            dialogue_path = item / "dialogue_transcription.txt"
            sounds_path = item / "music_caption.srt"

            dialogue = clean_lines(read_text_file(dialogue_path).splitlines())
            raw_sounds = read_text_file(sounds_path).splitlines()
            sounds = remove_consecutive_duplicates(clean_lines(raw_sounds))
            visual = extract_visual_descriptions_in_range_from_csv(
                start_time, end_time, csv_path)

            subtitles = clean_lines(extract_subtitles_in_range(
                global_subs, start_time, end_time).splitlines())

            output["scenes"].append({
                "scene_id": scene_id,
                "start_time": start_time,
                "end_time": end_time,
                "subtitles": subtitles,
                "dialogues": dialogue,
                "non_dialogue_sounds": sounds,
                "visual_description": visual
            })

    return output


def merge_data(movie_name):
    path = f"{RUTA_TEXTO}/{movie_name}"
    data = create_movie_json(path, movie_name)

    output_file = f"{RUTA_TEXTO}/{movie_name}/{movie_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Archivo guardado: {output_file}")

# if __name__ == "__main__":
#     movie_name = "Zootopia"
#     merge_data(movie_name)
