import json
from CONSTANTS import RUTA_DB

# Ruta al archivo original y de salida
def json2jsonl():
    
    # input_path = "scenes_data.json"
    input_path = f"{RUTA_DB}/RAG.json"
    output_path = f"{RUTA_DB}/RAG_DB.jsonl"

    # Cargar el JSON completo
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convertir a JSONL
    with open(output_path, "w", encoding="utf-8") as f_out:
        for movie_name, movie in data["movies"].items():
            metadata = movie["metadata"]
            scenes = movie["scenes"]
            for scene in scenes:
                scene_id = scene["scene_id"]

                # Concatenar contenido
                content = ""
                if "subtitles" in scene:
                    content += "\n".join(scene["subtitles"]) + "\n"
                if "dialogues" in scene:
                    content += "\n".join(scene["dialogues"]) + "\n"
                if "non_dialogue_sounds" in scene:
                    content += "\n".join(scene["non_dialogue_sounds"]) + "\n"
                if "visual_description" in scene:
                    content += "\n".join(scene["visual_description"])

                # Crear entrada JSONL
                entry = {
                    "id": f"{movie_name.lower()}_scene_{str(scene_id).zfill(3)}",
                    "movie": movie_name,
                    "metadata": metadata,
                    "start_time": scene.get("start_time"),
                    "end_time": scene.get("end_time"),
                    "content": content.strip()
                }

                # Escribir línea
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
