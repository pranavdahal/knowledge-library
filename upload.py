import pandas as pd
import numpy as np
from knowledge_graph import KnowledgeGraph
from knn_classifier import KNNClassifier
from nlp_classifier import NLPClassifier
from map_compliance import ComplianceMapper

import warnings
warnings.filterwarnings("ignore")

class UploadFile:
    def __init__(self):
        # self.knn_classifier = KNNClassifier()
        self.nlp_classifier = NLPClassifier()
        self.nlp_classifier.load_model()
        self.kg = KnowledgeGraph()
        self.compliance_mapper = ComplianceMapper()
        
    def handle_file_qna_pair(self, df: pd.DataFrame, client_name: str):
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        if 'question' in df.columns or 'question_text' in df.columns:
            questions = df['question'].tolist() if 'question' in df.columns else df['question_text'].tolist()

            # categories, subcategories = self.knn_classifier.classify_multiple_questions(questions)
            # df['category'] = categories
            # df['subcategory'] = subcategories

            classified_df = self.nlp_classifier.classify_bulk(questions)
            df['category'] = classified_df['category']
            df['subcategory'] = classified_df['subcategory']

        else:
            return False

        self.kg.add_qna_data_to_graph(df, client_name)
        self.compliance_mapper.map_qna_to_controls(client_name)
    
    def add_qna_pair(self,question, answer, note, client, category, subcategory):
        if subcategory == "Select Subcategory":
            if category == "Select Category":
                # category, subcategory = self.knn_classifier.classify_question(question)
                category, subcategory = self.nlp_classifier.classify(question)
            else:
                _, subcategory = self.nlp_classifier.classify(question, category)

        self.kg.add_qna_pair(question, answer,client, category, subcategory, note=note)

        self.compliance_mapper.map_qna_to_controls(client)
    

