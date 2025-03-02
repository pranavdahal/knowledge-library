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


class CategoryClassifier(nn.Module):
    def __init__(self, embedding_dim, num_categories):
        super(CategoryClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_categories)
        )
        
    def forward(self, x):
        return self.classifier(x)


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


class NLPClassifier:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.device = torch.device('cpu')
        self.category_classifier = None
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        

        num_categories = train_dataset.get_num_categories()
        self.category_classifier = CategoryClassifier(self.embedding_dim, num_categories)
        self.category_classifier = self.category_classifier.to(self.device)
        
        self.subcategory_classifier = SubCategoryClassifier(taxonomy_df)
        
        print(f"Training model on {self.device}...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.category_classifier.parameters(), lr=learning_rate)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Training phase
            self.category_classifier.train()
            train_loss = 0
            
            for embeddings, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.category_classifier(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.category_classifier.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for embeddings, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                    embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                    outputs = self.category_classifier(embeddings)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}')
            print(f'  Val Accuracy: {val_accuracy:.2f}%')
        
        print("Training completed!")
        return history
    
    def save_model(self, model_dir='./models/linear_nn', model_name='linear_nn_classifier'):
        if self.category_classifier is None or self.label_encoder is None:
            raise ValueError("Model or label encoder not initialized. Train the model first.")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save classifier
        model_path = os.path.join(model_dir, f"{model_name}_model.pt")
        torch.save(self.category_classifier.state_dict(), model_path)
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, f"{model_name}_encoder.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save taxonomy data
        if self.subcategory_classifier is not None:
            taxonomy_path = os.path.join(model_dir, f"{model_name}_taxonomy.pkl")
            joblib.dump(self.subcategory_classifier.keywords_df, taxonomy_path)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='./models/linear_nn', model_name='linear_nn_classifier', num_categories=None):
        encoder_path = os.path.join(model_dir, f"{model_name}_encoder.pkl")
        self.label_encoder = joblib.load(encoder_path)
        
        if num_categories is None:
            num_categories = len(self.label_encoder.classes_)
        
        self.category_classifier = CategoryClassifier(self.embedding_dim, num_categories)
        model_path = os.path.join(model_dir, f"{model_name}_model.pt")
        self.category_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        self.category_classifier = self.category_classifier.to(self.device)
        self.category_classifier.eval()
        
        taxonomy_path = os.path.join(model_dir, f"{model_name}_taxonomy.pkl")
        if os.path.exists(taxonomy_path):
            taxonomy_df = joblib.load(taxonomy_path)
            self.subcategory_classifier = SubCategoryClassifier(taxonomy_df)
    
    def classify(self, question, category=None):
        if self.category_classifier is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        embedding = self.embedding_model.encode(question)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)

        if category is not None:
            predicted_category = category
        else:
            # Predict category
            self.category_classifier.eval()
            with torch.no_grad():
                output = self.category_classifier(embedding_tensor)
                _, predicted_idx = torch.max(output, 1)
                predicted_category = self.label_encoder.inverse_transform([predicted_idx.item()])[0]
            
        # Predict subcategory
        predicted_subcategory = None
        if self.subcategory_classifier is not None:
            predicted_subcategory = self.subcategory_classifier.get_subcategory(
                embedding_tensor, predicted_category
            )
        
        return predicted_category, predicted_subcategory

    def classify_bulk(self, questions):

        if self.category_classifier is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for question in questions:
            result = {}
            predicted_category, predicted_subcategory = self.classify(question)
            result['category'] = predicted_category
            result['subcategory'] = predicted_subcategory
            results.append(result)

        
        return pd.DataFrame(results)