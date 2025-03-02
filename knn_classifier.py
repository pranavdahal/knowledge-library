from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import pandas as pd
import numpy as np
import json
import joblib
import os

class KNNClassifier:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def save_model(self, filepath="classification_model/knn_model.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            "knn_category": self.knn_category,
            "label_encoder": self.label_encoder,
        }

        joblib.dump(model_data, filepath)

    def train(self):
        # load generated questions with labels
        df = pd.read_csv('generated_data/all_questions.csv')

        self.label_encoder = LabelEncoder()
        df['category_label'] = self.label_encoder.fit_transform(df['category'])

        question_embeddings = self.embedding_model.encode(df['question'].tolist())

        self.knn_category = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        self.knn_category.fit(question_embeddings, df['category_label'])

        self.save_model()
    
    def load_model(self, filepath="classification_model/knn_model.pkl"):
        model_data = joblib.load(filepath)
        self.knn_category = model_data["knn_category"]
        self.label_encoder = model_data["label_encoder"]
    
    def predict_subcategory(self,question_embedding, predicted_category, embedding_model):
        taxonomy_df = pd.read_csv('generated_data/taxonomy_with_keywords.csv')
        category_data = taxonomy_df[taxonomy_df['category'] == predicted_category]
        
        max_similarity = -1
        predicted_subcategory = None
        
        for index, row in category_data.iterrows():
            subcategory = row['subcategory']
            keywords = row['keywords']
            
            keywords_embedding = embedding_model.encode(keywords)
            
            similarity = cosine_similarity(question_embedding, keywords_embedding.reshape(1,-1))[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_subcategory = subcategory
        
        return predicted_subcategory
        
    def classify_question(self, question,category=None):
        self.load_model()
        q_embed = self.embedding_model.encode([question])
        if category is not None:
            predicted_category = category
        else:
            predicted_category = self.knn_category.predict(q_embed)
            predicted_category = self.label_encoder.inverse_transform(predicted_category)
            predicted_category = predicted_category[0]

        predicted_subcategory = self.predict_subcategory(q_embed, predicted_category, self.embedding_model)
        
        return predicted_category, predicted_subcategory
    
    def classify_multiple_questions(self, questions):
        self.load_model()
        q_embed = self.embedding_model.encode(questions)

        predicted_subcategories = []
        predicted_categories = []
        for i in range(len(questions)):
            predicted_category = self.knn_category.predict(q_embed)
            predicted_category = self.label_encoder.inverse_transform(predicted_category)
            predicted_categories.append(predicted_category[0])

            predicted_subcategory = self.predict_subcategory(q_embed[i].reshape(1,-1), predicted_category[0], self.embedding_model)
            predicted_subcategories.append(predicted_subcategory)
        
        return predicted_categories, predicted_subcategories

