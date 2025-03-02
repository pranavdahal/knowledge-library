import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from knowledge_graph import KnowledgeGraph
import json
import os

import warnings
warnings.filterwarnings("ignore")

class ComplianceMapper():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_graph = KnowledgeGraph()
        self.controls = self.knowledge_graph.get_all_controls()
    
    def import_compliance_standard(self, json_file_path):
        try:
            with open(json_file_path, 'r') as file:
                compliance_data = json.load(file)
            
            standard_name = compliance_data.get('standard')
            standard_version = compliance_data.get('version')
            
            if not standard_name:
                return False
            
            self.knowledge_graph.create_standard(standard_name, standard_version)
            
            domains = compliance_data.get('domains', [])
            for domain in domains:
                domain_id = domain.get('domain_id')
                domain_name = domain.get('domain_name')
                
                if not domain_id or not domain_name:
                    continue
                    
                self.knowledge_graph.create_domain(domain_id, domain_name, standard_name, standard_version)
                
                controls = domain.get('controls', [])
                for control in controls:
                    control_id = control.get('control_id')
                    title = control.get('title')
                    description = control.get('description')
                    
                    if not control_id or not title:
                        continue
                    
                    self.knowledge_graph.create_control(
                        control_id, 
                        title, 
                        description,
                        domain_id,
                        domain_name
                    )
            
            for domain in domains:
                controls = domain.get('controls', [])
                for control in controls:
                    control_id = control.get('control_id')
                    related_controls = control.get('related_controls', [])
                    
                    for related_id in related_controls:
                        self.knowledge_graph.create_control_relationship(control_id, related_id)
            
            return True
            
        except Exception as e:
            return False
        
    def import_all_standards(self, directory_path='new_generated_data/compliance_standards'):
        count = 0
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                if self.import_compliance_standard(file_path):
                    count += 1
        return count
    
    def map_qna_to_controls(self, client_name):
        qna_pairs = self.knowledge_graph.get_qna_pairs_client(client_name)
        if len(self.controls) == 0:
            print("Importing all standards")
            self.import_all_standards()
            self.controls = self.knowledge_graph.get_all_controls()

        control_texts = [f"{control['control_name']} {control['domain_name']} {control['description']}" for control in self.controls]
        control_embeddings = self.model.encode(control_texts)

        tfidf = TfidfVectorizer(stop_words='english')
        control_tfidf = tfidf.fit_transform([f"{c['control_name']} {c['domain_name']} {c['description']}" for c in self.controls])

        for qna in qna_pairs:
            qna_text = qna['question']
            qna_embedding = self.model.encode([qna_text])[0]
            
            semantic_similarities = cosine_similarity([qna_embedding], control_embeddings)[0]
            
            qna_tfidf = tfidf.transform([qna_text])
            keyword_similarities = cosine_similarity(qna_tfidf, control_tfidf)[0]
            
            
            combined_similarities = 0.7 * semantic_similarities + 0.3 * keyword_similarities
            top_indices = np.argsort(combined_similarities)[-10:][::-1]
            
            best_match_index = None
            best_match_score = 0.0
            
            for idx in top_indices:
                control = self.controls[idx]
                control_text = f"{control['control_name']} {control['description']} {control.get('guidance', '')}"
                
                sim_score = combined_similarities[idx]

                if sim_score > best_match_score:
                    best_match_score = sim_score
                    best_match_index = idx
            
            if best_match_score > 0.3:
                best_control = self.controls[best_match_index]
                self.knowledge_graph.update_qna_compliance_mapping(qna['id'],  best_control['id'], best_control['control_name'], best_match_score)
            else:
                print(f"No good match found for QnA {qna['id']} (best score: {best_match_score:.4f})")
    
        






