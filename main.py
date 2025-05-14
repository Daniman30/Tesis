from time import time
from tools import elegir_mkv, segundos_a_hora
from CONSTANTS import RUTA_VIDEO
# from Code.RAG.rag import rag
from RAG.merge_data import merge_data
from sound2text.sound2text import sound2text
from video2text.video2text import video2text
from video2text.frame_extraction import frame_extraction


def main():
    name = elegir_mkv(RUTA_VIDEO)
    if name == '': return

    start = time()

    print("TRANSFORMANDO SONIDO A TEXTO")
    sound2text(name)

    time1 = time()
    s2t = time1 - start

    print("EXTRAYENDO FRAMES")
    frame_extraction(name)

    time2 = time()
    fe = time2 - time1

    print("TRANSFORMANDO VIDEO A TEXTO")
    video2text(name)

    time3 = time()
    v2t = time3 - time2

    print("MEZCLANDO LA INFORMACION")
    merge_data(name)

    time4 = time()
    md = time4 - time3
    total = time4 - start

    print(f"EN TOTAL SE DEMORO: {segundos_a_hora(total)} ")
    print(f"EN TRANSFORMAR SONIDO A TEXTO SE DEMORO: {segundos_a_hora(s2t)}")
    print(f"EN EXTRAER FRAMES SE DEMORO: {segundos_a_hora(fe)}")
    print(f"EN TRANSFORMAR VIDEO A TEXTO SE DEMORO: {segundos_a_hora(v2t)}")
    print(f"EN MEZCLAR LA INFO SE DEMORO: {segundos_a_hora(md)}")


if __name__ == "__main__":
    main()



#! COMBINAR LOS JSON EN UN SOLO ARCHIVO EN LA RUTA CORRECTA
#! CREAR RAG
#! TESTING