import json
import torch
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import warnings
from CONSTANTS import RUTA_DB
warnings.filterwarnings("ignore")


class AdvancedRAGSystem:
    def __init__(self):
        # Cargar modelo de embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Cargar modelo generativo
        self.generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

        # Modelo de reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Vectorizador TF-IDF para análisis de relevancia
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Inicializar estructuras de datos
        self.documents = []
        self.embeddings = []
        self.index = None
        self.tfidf_matrix = None

        self.load_documents()

    def get_embedding(self, text):
        """Generar embeddings para texto"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def load_documents(self):
        """Cargar documentos y generar embeddings"""
        print("Cargando documentos y generando embeddings...")

        file_path = f"{RUTA_DB}/RAG_DB.jsonl"

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.documents.append(doc)
                emb = self.get_embedding(doc['content'])
                self.embeddings.append(emb)

        # Crear índice FAISS
        dimension = self.embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings))

        # Crear matriz TF-IDF
        doc_texts = [doc['content'] for doc in self.documents]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)

        print(f"Cargados {len(self.documents)} documentos")

    def retrieve_documents(self, question, top_k=10):
        """Recuperar documentos relevantes"""
        q_emb = self.get_embedding(question)
        distances, indices = self.index.search(np.array([q_emb]), top_k)

        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            retrieved_docs.append({
                'content': self.documents[idx]['content'],
                'index': idx,
                'distance': distances[0][i],
                'metadata': self.documents[idx].get('metadata', {})
            })

        return retrieved_docs

    def rerank_documents(self, question, retrieved_docs, top_k=5):
        """Reranking usando CrossEncoder - MEJORA LA SELECCIÓN DE CONTEXTO"""
        print("Aplicando reranking para mejorar selección de contexto...")

        # Preparar pares pregunta-documento para reranking
        pairs = [(question, doc['content']) for doc in retrieved_docs]

        # Obtener puntuaciones de reranking
        scores = self.reranker.predict(pairs)

        # Combinar documentos con puntuaciones y reordenar
        for i, doc in enumerate(retrieved_docs):
            doc['rerank_score'] = float(scores[i])

        # Ordenar por puntuación de reranking y tomar top_k
        reranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_docs[:top_k]

    def extract_and_weight_relevant_passages(self, question, documents):
        """Extraer y ponderar pasajes relevantes - MEJORA EL CONTEXTO"""
        print("Extrayendo pasajes más relevantes para mejorar contexto...")

        question_words = set(question.lower().split())
        question_keywords = [word for word in question_words if len(word) > 3]

        weighted_passages = []

        for doc in documents:
            content = doc['content']
            sentences = re.split(r'[.!?]+', content)

            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 20:  # Filtrar oraciones muy cortas
                    continue

                sentence_lower = sentence.lower()
                sentence_words = set(sentence_lower.split())

                # Calcular relevancia
                keyword_overlap = len(set(question_keywords).intersection(sentence_words))
                word_overlap = len(question_words.intersection(sentence_words))

                # Peso por posición (primeras oraciones tienen más peso)
                position_weight = 1.5 if i < 3 else 1.0

                # Peso por longitud (oraciones más informativas)
                length_weight = min(2.0, len(sentence.split()) / 15.0)

                relevance_score = (keyword_overlap * 2 + word_overlap) * position_weight * length_weight

                if relevance_score > 1.0:  # Umbral de relevancia
                    weighted_passages.append({
                        'text': sentence.strip(),
                        'relevance_score': relevance_score,
                        'source_doc': doc['index'],
                        'rerank_score': doc.get('rerank_score', 0)
                    })

        # Ordenar por relevancia y tomar los mejores
        weighted_passages.sort(key=lambda x: x['relevance_score'] * (1 + x['rerank_score']), reverse=True)
        return weighted_passages[:15]  # Top 15 pasajes más relevantes

    def detect_and_resolve_contradictions(self, passages, question):
        """Detectar contradicciones y resolverlas - MEJORA LA COHERENCIA"""
        print("Detectando y resolviendo contradicciones para mejorar coherencia...")

        contradictions = []
        negation_words = ['not', 'no', 'never', 'neither', 'nor',
                          'cannot', 'isn\'t', 'aren\'t', 'won\'t', 'don\'t']

        # Detectar contradicciones
        for i, p1 in enumerate(passages):
            for j, p2 in enumerate(passages[i+1:], i+1):
                text1 = p1['text'].lower()
                text2 = p2['text'].lower()

                words1 = set(text1.split())
                words2 = set(text2.split())
                common_words = words1.intersection(words2)

                if len(common_words) >= 3:
                    has_negation1 = any(neg in text1 for neg in negation_words)
                    has_negation2 = any(neg in text2 for neg in negation_words)

                    if has_negation1 != has_negation2:  # Contradicción detectada
                        contradiction_strength = len(
                            common_words) / max(len(words1), len(words2))

                        contradictions.append({
                            'index1': i,
                            'index2': j,
                            'strength': contradiction_strength,
                            'passage1': p1,
                            'passage2': p2
                        })

        # Resolver contradicciones: mantener el pasaje con mayor relevancia
        indices_to_remove = set()
        for contradiction in contradictions:
            if contradiction['strength'] > 0.3:  # Contradicción significativa
                p1_score = contradiction['passage1']['relevance_score']
                p2_score = contradiction['passage2']['relevance_score']

                # Remover el pasaje con menor puntuación
                if p1_score < p2_score:
                    indices_to_remove.add(contradiction['index1'])
                else:
                    indices_to_remove.add(contradiction['index2'])

        # Filtrar pasajes contradictorios
        filtered_passages = [p for i, p in enumerate(
            passages) if i not in indices_to_remove]

        return filtered_passages, len(contradictions)

    def apply_attention_weighting(self, passages, question):
        """Aplicar pesos de atención para mejorar el contexto final"""
        print("Aplicando pesos de atención para optimizar contexto...")

        question_words = set(question.lower().split())

        for passage in passages:
            text = passage['text'].lower()
            words = set(text.split())

            # Calcular atención basada en overlap semántico
            semantic_attention = len(question_words.intersection(
                words)) / len(words.union(question_words))

            # Atención por frecuencia de palabras clave
            keyword_attention = sum(
                1 for word in question_words if word in text) / len(question_words)

            # Combinar atenciones
            total_attention = (semantic_attention * 0.6 +
                               keyword_attention * 0.4)
            passage['attention_weight'] = total_attention

            # Ajustar puntuación de relevancia con atención
            passage['final_score'] = passage['relevance_score'] * \
                (1 + total_attention)

        # Reordenar por puntuación final
        passages.sort(key=lambda x: x['final_score'], reverse=True)
        return passages

    def build_optimized_context(self, passages, max_length=800):
        """Construir contexto optimizado usando toda la información procesada"""
        print("Construyendo contexto optimizado...")

        # Tomar los mejores pasajes hasta alcanzar la longitud máxima
        context_parts = []
        current_length = 0

        for passage in passages:
            text = passage['text']
            if current_length + len(text.split()) <= max_length:
                # Añadir peso de confianza al texto si es muy relevante
                if passage['final_score'] > 2.0:
                    context_parts.append(f"[HIGH RELEVANCE] {text}")
                else:
                    context_parts.append(text)
                current_length += len(text.split())
            else:
                break

        return " ".join(context_parts)

    def self_verify_and_enhance_generation(self, question, initial_context, passages):
        """Auto-verificación que mejora la generación"""
        print("Aplicando auto-verificación para mejorar generación...")

        # Verificar cobertura de la pregunta
        question_words = set(question.lower().split())
        context_words = set(initial_context.lower().split())
        coverage = len(question_words.intersection(
            context_words)) / len(question_words)

        enhanced_context = initial_context

        # Si la cobertura es baja, añadir más contexto relevante
        if coverage < 0.5:
            print("Cobertura baja detectada, añadiendo contexto adicional...")
            additional_passages = [p for p in passages[len(initial_context.split()):]
                                   if any(word in p['text'].lower() for word in question_words)]

            if additional_passages:
                additional_text = " ".join(
                    [p['text'] for p in additional_passages[:3]])
                enhanced_context = f"{initial_context} [ADDITIONAL CONTEXT] {additional_text}"

        # Añadir instrucciones específicas para mejorar la respuesta
        instruction_prefix = self._generate_instruction_prefix(
            question, passages)

        return f"{instruction_prefix} {enhanced_context} {question}"

    def _generate_instruction_prefix(self, question, passages):
        """Generar prefijo de instrucciones basado en el análisis"""

        # Determinar tipo de pregunta
        question_lower = question.lower()
        if any(word in question_lower for word in ['how', 'How', 'HOW']):
            instruction = "Provide a detailed step-by-step explanation."
        elif any(word in question_lower for word in ['why', 'Why', 'WHY']):
            instruction = "Explain the reasons and causes comprehensively."
        elif any(word in question_lower for word in ['what', 'What', 'WHAT']):
            instruction = "Define and describe thoroughly."
        elif any(word in question_lower for word in ['when', 'When', 'WHEN']):
            instruction = "Specify timing and chronological details."
        else:
            instruction = "Provide a comprehensive and accurate answer."

        # Añadir instrucciones sobre calidad si hay muchos pasajes relevantes
        if len(passages) > 10:
            instruction += " Synthesize information from multiple sources."

        return f"[INSTRUCTION] {instruction}"

    def calculate_response_confidence(self, question, context, passages, contradictions_count):
        """Calcular confianza para ajustar parámetros de generación"""

        # Factores de confianza
        context_coverage = len(set(question.lower().split()).intersection(
            set(context.lower().split()))) / len(set(question.lower().split()))
        passage_quality = np.mean([p['final_score']for p in passages[:5]]) if passages else 0
        contradiction_penalty = max(0, 1 - contradictions_count * 0.1)

        confidence = (context_coverage * 0.4 + min(1.0,
                      passage_quality / 3.0) * 0.4 + contradiction_penalty * 0.2)

        return confidence

    def answer_question(self, question, top_k_retrieve=10, top_k_rerank=5):
        """Función principal que integra todas las mejoras para generar mejor respuesta"""
        print(
            f"\n=== Procesando pregunta con mejoras integradas: {question} ===")

        # 1. Recuperar documentos iniciales
        retrieved_docs = self.retrieve_documents(question, top_k_retrieve)
        print("~"*80)
        print("RETRIEVE DOCS")
        print(retrieved_docs)
        print("~"*80)

        # 2. MEJORA 1: Reranking para mejor selección
        reranked_docs = self.rerank_documents(
            question, retrieved_docs, top_k_rerank)
        print("~"*80)
        print("RERANKED DOCS")
        print(reranked_docs)
        print("~"*80)

        # 3. MEJORA 2: Extraer y ponderar pasajes relevantes
        relevant_passages = self.extract_and_weight_relevant_passages(
            question, reranked_docs)
        print("~"*80)
        print("RELEVANT PASSAGES")
        print(relevant_passages)
        print("~"*80)

        # 4. MEJORA 3: Detectar y resolver contradicciones
        filtered_passages, contradictions_count = self.detect_and_resolve_contradictions(
            relevant_passages, question)
        print(f"Contradicciones resueltas: {contradictions_count}")
        print("~"*80)
        print("FILTERED PASSAGES")
        print(filtered_passages)
        print("~"*80)

        # 5. MEJORA 4: Aplicar pesos de atención
        attention_weighted_passages = self.apply_attention_weighting(
            filtered_passages, question)
        print("~"*80)
        print("ATTENTION WEIGHTED PASSAGES")
        print(attention_weighted_passages)
        print("~"*80)

        # 6. MEJORA 5: Construir contexto optimizado
        optimized_context = self.build_optimized_context(
            attention_weighted_passages)
        print("~"*80)
        print("OPTIMIZED CONTEXT")
        print(optimized_context)
        print("~"*80)

        # 7. MEJORA 6: Auto-verificación y mejora de la entrada
        enhanced_input = self.self_verify_and_enhance_generation(
            question, optimized_context, attention_weighted_passages)
        print("~"*80)
        print("ENHENCED INPUT")
        print(enhanced_input)
        print("~"*80)

        # 8. Calcular confianza para ajustar generación
        confidence = self.calculate_response_confidence(
            question, optimized_context, attention_weighted_passages, contradictions_count)
        print("~"*80)
        print("CONFIDENCE")
        print(confidence)
        print("~"*80)

        # 9. Generar respuesta con parámetros ajustados por confianza
        inputs = self.generator_tokenizer.encode(
            enhanced_input, return_tensors='pt', truncation=True, max_length=10240)
        print("~"*80)
        print("INPUTS")
        print(inputs)
        print("~"*80)

        # Ajustar parámetros de generación basados en confianza
        # Más conservador si alta confianza
        temperature = 0.3 if confidence > 0.8 else 0.7
        num_beams = 7 if confidence > 0.7 else 5

        with torch.no_grad():
            outputs = self.generator_model.generate(
                inputs,
                max_length=2500,
                num_beams=num_beams,
                early_stopping=True,
                temperature=temperature,
                do_sample=True if temperature > 0.5 else False,
                top_p=0.9 if temperature > 0.5 else None
            )

        answer = self.generator_tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        print("\n" + "="*80)
        print("RESPUESTA MEJORADA:")
        print("="*80)
        print(f"Pregunta: {question}")
        print(f"Respuesta: {answer}")
        print(f"Confianza del sistema: {confidence:.2f}")
        print(f"Pasajes procesados: {len(attention_weighted_passages)}")
        print(f"Contradicciones resueltas: {contradictions_count}")
        print("="*80)

        return {
            'answer': answer,
            'confidence': confidence,
            'processed_passages': len(attention_weighted_passages),
            'contradictions_resolved': contradictions_count,
            'context_length': len(optimized_context.split())
        }
