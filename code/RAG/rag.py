import json
import torch
import faiss
import numpy as np
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# Cargar modelo de embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Función para generar embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Cargar corpus y generar embeddings
documents = []
embeddings = []
with open('D:/UNI/Tesis/DB/RAG_DB.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line)
        documents.append(doc)
        emb = get_embedding(doc['content'])
        embeddings.append(emb)

# Crear índice FAISS
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Cargar modelo generativo
generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Función para responder preguntas
def answer_question(question, top_k=5):
    # Generar embedding de la pregunta
    q_emb = get_embedding(question)
    # Recuperar documentos relevantes
    distances, indices = index.search(np.array([q_emb]), top_k)
    retrieved_docs = [documents[i]['content'] for i in indices[0]]
    # Concatenar contexto
    context = " ".join(retrieved_docs)
    print(context)
    # Preparar entrada para el generador
    input_text = f"{question}.\n \n Just answer the specific Question in a few words with the following context. \n \n {context}"
    inputs = generator_tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=1024)
    # Generar respuesta
    outputs = generator_model.generate(inputs, max_length=200, num_beams=5, early_stopping=True)
    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("ANSWER: ")
    return answer

# Ejemplo de uso
question = "Who is Mark in Jurassic Park III?"
print(answer_question(question))

