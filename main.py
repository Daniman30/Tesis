from time import time
from tools import elegir_mkv, segundos_a_hora
from CONSTANTS import RUTA_VIDEO
from RAG.merge_data import merge_data
from RAG.process_data import json2jsonl
from RAG.rag2 import AdvancedRAGSystem
from video2text.video2text import video2text
from sound2text.sound2text import sound2text
from video2text.frame_extraction import frame_extraction

def preprocess():
    # PREPROCESAMIENTO
    name = elegir_mkv(RUTA_VIDEO)
    if name == '': return

    start = time()

    print("TRANSFORMANDO SONIDO A TEXTO")
    sound2text(name, True, 120)

    time1 = time()
    s2t = time1 - start
    print(f"EN TRANSFORMAR SONIDO A TEXTO SE DEMORO: {segundos_a_hora(s2t)}")

    print("EXTRAYENDO FRAMES")
    frame_extraction(name)

    time2 = time()
    fe = time2 - time1
    print(f"EN EXTRAER FRAMES SE DEMORO: {segundos_a_hora(fe)}")

    print("TRANSFORMANDO VIDEO A TEXTO")
    video2text(name)

    time3 = time()
    v2t = time3 - time2
    print(f"EN TRANSFORMAR VIDEO A TEXTO SE DEMORO: {segundos_a_hora(v2t)}")

    print("MEZCLANDO LA INFORMACION")
    merge_data(name)
    
    json2jsonl()

    time4 = time()
    md = time4 - time3
    total = time4 - start

    print(f"EN TOTAL SE DEMORO: {segundos_a_hora(total)} ")
    print(f"EN TRANSFORMAR SONIDO A TEXTO SE DEMORO: {segundos_a_hora(s2t)}")
    print(f"EN EXTRAER FRAMES SE DEMORO: {segundos_a_hora(fe)}")
    print(f"EN TRANSFORMAR VIDEO A TEXTO SE DEMORO: {segundos_a_hora(v2t)}")
    print(f"EN MEZCLAR LA INFO SE DEMORO: {segundos_a_hora(md)}")

def RAG():
    # USO DE RAG
    question = input()
    #"Who is Mark in Jurassic Park III?"
    
    rag_system = AdvancedRAGSystem()
    print(rag_system.answer_question(question))
    

def main():
    # preprocess()
    
    RAG()
    


if __name__ == "__main__":
    main()

