import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieQAEvaluator:
    def __init__(self, dataset_path):
        """Carga el dataset MovieQA en formato JSON"""
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        
        # División del dataset según el paper
        self.val_data = [item for item in self.data if item['split'] == 'val']
        
    def _classify_question(self, question):
        """Clasifica la pregunta por tipo (Who, What, Why, How, etc.)"""
        question = question.lower()
        if question.startswith('why'):
            return 'Why'
        elif question.startswith('how'):
            return 'How'
        elif question.startswith('who'):
            return 'Who'
        elif question.startswith('what'):
            return 'What'
        elif question.startswith('where'):
            return 'Where'
        elif question.startswith('when'):
            return 'When'
        else:
            return 'Other'
    
    def evaluate_complex_questions(self, rag_system):
        """Evaluación 1.3: Análisis de preguntas complejas vs factuales"""
        results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for item in self.val_data:
            question_type = self._classify_question(item['question'])
            prediction = rag_system.predict(
                question=item['question'],
                context=item['context']
            )
            
            results[question_type]['total'] += 1
            if prediction == item['correct_answer']:
                results[question_type]['correct'] += 1
        
        # Calcula precisión por tipo
        accuracies = {
            qtype: info['correct']/info['total'] 
            for qtype, info in results.items()
        }
        
        print("Precisión por tipo de pregunta:")
        for qtype, acc in accuracies.items():
            print(f"{qtype}: {acc:.2%}")
        
        return accuracies
    
    def evaluate_deceptive_answers(self, rag_system):
        """Evaluación 1.4: Robustez ante respuestas engañosas"""
        deceptive_analysis = {
            'length_analysis': {'correct_longer': 0, 'total': 0},
            'similarity_analysis': {'correct_most_similar': 0, 'total': 0}
        }
        
        vectorizer = TfidfVectorizer()
        
        for item in self.val_data:
            all_answers = [item[f'answer_{i}'] for i in range(5)]
            correct_idx = all_answers.index(item['correct_answer'])
            
            # Análisis de longitud
            lengths = [len(ans.split()) for ans in all_answers]
            max_length_idx = np.argmax(lengths)
            if max_length_idx == correct_idx:
                deceptive_analysis['length_analysis']['correct_longer'] += 1
            deceptive_analysis['length_analysis']['total'] += 1
            
            # Análisis de similitud con la pregunta
            tfidf_matrix = vectorizer.fit_transform([item['question']] + all_answers)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:6])
            most_similar_idx = np.argmax(similarities[0])
            
            if most_similar_idx == correct_idx:
                deceptive_analysis['similarity_analysis']['correct_most_similar'] += 1
            deceptive_analysis['similarity_analysis']['total'] += 1
            
        # Resultados
        print("\nAnálisis de respuestas engañosas:")
        print(f"Respuesta correcta era la más larga: {deceptive_analysis['length_analysis']['correct_longer']/deceptive_analysis['length_analysis']['total']:.2%}")
        print(f"Respuesta correcta más similar a la pregunta: {deceptive_analysis['similarity_analysis']['correct_most_similar']/deceptive_analysis['similarity_analysis']['total']:.2%}")
        
        return deceptive_analysis
    
    def compare_baselines(self, rag_system):
        """Evaluación 1.5: Comparación con baselines del paper"""
        # Baseline 1: Hasty Student (longest answer)
        hasty_correct = 0
        
        # Baseline 2: Searching Student (cosine similarity)
        cosine_correct = 0
        
        total = len(self.val_data)
        
        for item in self.val_data:
            all_answers = [item[f'answer_{i}'] for i in range(5)]
            
            # Hasty Student: elige la respuesta más larga
            lengths = [len(ans.split()) for ans in all_answers]
            hasty_prediction = all_answers[np.argmax(lengths)]
            if hasty_prediction == item['correct_answer']:
                hasty_correct += 1
                
            # Searching Student: similitud con contexto
            context = item['context']
            question = item['question']
            
            # Implementación simplificada de cosine similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([context] + [question] + all_answers)
            context_vec = tfidf_matrix[0]
            question_vec = tfidf_matrix[1]
            answer_vecs = tfidf_matrix[2:7]
            
            similarities = cosine_similarity( 
                question_vec, 
                np.vstack([context_vec.toarray(), answer_vecs.toarray()])
            )
            
            cosine_prediction = all_answers[np.argmax(similarities[0][1:])]
            if cosine_prediction == item['correct_answer']:
                cosine_correct += 1
                
        # Evaluación del sistema RAG
        rag_correct = 0
        for item in self.val_data:
            prediction = rag_system.predict(
                question=item['question'],
                context=item['context']
            )
            if prediction == item['correct_answer']:
                rag_correct += 1
                
        # Resultados
        print("\nComparación con baselines:")
        print(f"Hasty Student (longest): {hasty_correct/total:.2%}")
        print(f"Searching Student (cosine): {cosine_correct/total:.2%}")
        print(f"Sistema RAG: {rag_correct/total:.2%}")
        
        return {
            'hasty': hasty_correct/total,
            'cosine': cosine_correct/total,
            'rag': rag_correct/total
        }

# Ejemplo de uso:
class RAGSystem:
    def predict(self, question, context):
        """Implementación de ejemplo - reemplazar con tu sistema RAG"""
        # Aquí iría la lógica de tu sistema RAG
        # Este ejemplo retorna una respuesta aleatoria
        import random
        return random.choice(context.split('.')[:5])  # Simulación básica

if __name__ == "__main__":
    evaluator = MovieQAEvaluator("movieqa_val.json")
    rag = RAGSystem()
    
    # Ejecutar todas las evaluaciones
    evaluator.evaluate_complex_questions(rag)
    evaluator.evaluate_deceptive_answers(rag)
    evaluator.compare_baselines(rag)