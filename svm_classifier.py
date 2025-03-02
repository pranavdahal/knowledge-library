import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from tqdm import tqdm
import joblib
import os

class QADataset(Dataset):
    def __init__(self, questions, categories, model_name='all-MiniLM-L6-v2'):
        self.questions = questions
        self.model = SentenceTransformer(model_name)
        self.categories = categories
        self.label_encoder = LabelEncoder()
        self.encoded_categories = self.label_encoder.fit_transform(categories)
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        embedding = self.model.encode(question)
        category = self.encoded_categories[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(category, dtype=torch.long)
    
    def get_num_categories(self):
        return len(self.label_encoder.classes_)
    
    def decode_category(self, encoded_category):
        return self.label_encoder.inverse_transform([encoded_category])[0]

class SVMClassifier:
    def __init__(self):
        self.svm = SVC(kernel='linear', probability=True)
        self.is_trained = False
        
    def train(self, embeddings, labels):
        self.svm.fit(embeddings, labels)
        self.is_trained = True
        
    def predict(self, embeddings):
        if not self.is_trained:
            raise ValueError("SVM not trained yet")
        return self.svm.predict(embeddings)
    
    def predict_proba(self, embeddings):
        if not self.is_trained:
            raise ValueError("SVM not trained yet")
        return self.svm.predict_proba(embeddings)


class SubCategoryClassifier:
    def __init__(self, taxonomy_df, embedding_model_name='all-MiniLM-L6-v2'):
        self.keywords_df = taxonomy_df
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def get_subcategory(self, question_embedding, category):
        if category not in self.keywords_df['category'].values:
            return "Category not classified"

        category_data = self.keywords_df[self.keywords_df['category'] == category]
        
        max_similarity = -1
        predicted_subcategory = None
        
        for _, row in category_data.iterrows():
            subcategory = row['subcategory']
            keywords = row['keywords']
            description = row['description']

            combined_text = f"{subcategory} {description} keywords: {keywords}"

            combined_embedding = self.embedding_model.encode(combined_text)

            similarity = cosine_similarity(
                question_embedding.numpy(), 
                combined_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_subcategory = subcategory
        
        return predicted_subcategory


class NLPClassifierSVM:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.device = torch.device('cpu')
        self.category_classifier = SVMClassifier()
        self.label_encoder = None
        self.subcategory_classifier = None
        
    def train(self, questions_df, taxonomy_df, epochs=10, batch_size=32, learning_rate=1e-3, test_size=0.2, random_state=42):
        questions = questions_df['question'].tolist()
        categories = questions_df['category'].tolist()
        
        X_train, X_val, y_train, y_val = train_test_split(
            questions, categories, test_size=test_size, random_state=random_state
        )
        
        train_dataset = QADataset(X_train, y_train, self.embedding_model_name)
        val_dataset = QADataset(X_val, y_val, self.embedding_model_name)
        
        self.label_encoder = train_dataset.label_encoder
        
        print("Extracting embeddings for training...")
        train_embeddings = []
        train_labels = []
        
        for i in tqdm(range(len(train_dataset))):
            embedding, label = train_dataset[i]
            train_embeddings.append(embedding.numpy())
            train_labels.append(label.item())
        
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)
        
        print("Extracting embeddings for validation...")
        val_embeddings = []
        val_labels = []
        
        for i in tqdm(range(len(val_dataset))):
            embedding, label = val_dataset[i]
            val_embeddings.append(embedding.numpy())
            val_labels.append(label.item())
        
        val_embeddings = np.array(val_embeddings)
        val_labels = np.array(val_labels)
        
        self.subcategory_classifier = SubCategoryClassifier(taxonomy_df)
        
        print(f"Training SVM classifier...")
        self.category_classifier.train(train_embeddings, train_labels)
        
        val_predictions = self.category_classifier.predict(val_embeddings)
        val_accuracy = (val_predictions == val_labels).mean() * 100
        
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        print("Training completed!")
        return {'val_accuracy': val_accuracy}
    
    def save_model(self, model_dir='./models/svm', model_name='svm_classifier'):
        if not self.category_classifier.is_trained or self.label_encoder is None:
            raise ValueError("Model or label encoder not initialized. Train the model first.")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save SVM classifier
        model_path = os.path.join(model_dir, f"{model_name}_svm.pkl")
        joblib.dump(self.category_classifier.svm, model_path)
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, f"{model_name}_encoder.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save taxonomy data
        if self.subcategory_classifier is not None:
            taxonomy_path = os.path.join(model_dir, f"{model_name}_taxonomy.pkl")
            joblib.dump(self.subcategory_classifier.keywords_df, taxonomy_path)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='./models/svm', model_name='svm_classifier'):
        encoder_path = os.path.join(model_dir, f"{model_name}_encoder.pkl")
        self.label_encoder = joblib.load(encoder_path)
        
        # Load SVM classifier
        model_path = os.path.join(model_dir, f"{model_name}_svm.pkl")
        svm_model = joblib.load(model_path)
        self.category_classifier = SVMClassifier()
        self.category_classifier.svm = svm_model
        self.category_classifier.is_trained = True
        
        taxonomy_path = os.path.join(model_dir, f"{model_name}_taxonomy.pkl")
        if os.path.exists(taxonomy_path):
            taxonomy_df = joblib.load(taxonomy_path)
            self.subcategory_classifier = SubCategoryClassifier(taxonomy_df)
    
    def classify(self, question, category=None):
        if not self.category_classifier.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        embedding = self.embedding_model.encode(question)
        embedding_array = np.array([embedding])

        if category is not None:
            predicted_category = category
        else:
            # Predict category
            predicted_idx = self.category_classifier.predict(embedding_array)[0]
            predicted_category = self.label_encoder.inverse_transform([predicted_idx])[0]
            
        # Predict subcategory
        predicted_subcategory = None
        if self.subcategory_classifier is not None:
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            predicted_subcategory = self.subcategory_classifier.get_subcategory(
                embedding_tensor, predicted_category
            )
        
        return predicted_category, predicted_subcategory

    def classify_bulk(self, questions):
        if not self.category_classifier.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for question in questions:
            result = {}
            predicted_category, predicted_subcategory = self.classify(question)
            result['category'] = predicted_category
            result['subcategory'] = predicted_subcategory
            results.append(result)

        return pd.DataFrame(results)