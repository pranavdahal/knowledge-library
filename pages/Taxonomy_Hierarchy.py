import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

from knowledge_graph import KnowledgeGraph

st.set_page_config(page_title="Taxonomy Hierarchy", layout="wide")

class Hierarchy:
    def __init__(self):
        pass

    def display_hierarchy(self,chart_type):
        taxonomy = json.load(open('new_generated_data/taxonomy.json'))

        labels = []
        parents = []
        values = []

        for category in taxonomy.keys():
            labels.append(category)
            parents.append("Taxonomy")
            subcategories = taxonomy[category].get('subcategories', {})
            values.append(len(subcategories))

            for subcategory_details in subcategories:
                subcategory = subcategory_details['name']
                labels.append(subcategory)
                parents.append(category)
                values.append(1)
                

        labels.append("Taxonomy")
        parents.append("")
        values.append(len(taxonomy))

        if chart_type == "Sunburst":
            fig = go.Figure(go.Sunburst(
                labels=labels,
                parents=parents,
                # values=values,
                insidetextorientation='radial',
                maxdepth=2,
                textfont=dict(size=18, family="Arial, sans-serif"),
                marker=dict(
                    line=dict(width=1, color='black')
                ),
            ))

            fig.update_layout(
                margin=dict(t=0, b=5, r=5, l=5), 
                height=600, 
                width=700,
            )

        elif chart_type == "Treemap":
            fig = go.Figure(go.Treemap(
                labels=labels,
                parents=parents,
                # values=values,
                textfont=dict(size=20, family="Arial, sans-serif"),
                marker=dict(
                    line=dict(width=2, color='white')
                ),
                maxdepth=2
            ))

            fig.update_layout(
                margin=dict(t=0, b=10, r=10, l=10),
                height=600,
                width=700,
            )

        st.plotly_chart(fig)

class HierarchyAnalysis:
    def __init__(self):
        self.driver = KnowledgeGraph().driver
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_similarities(self,df,similarity_matrix, groupby_col):
        intra_group_similarities = {}
        inter_group_similarities = {}

        labels = df[groupby_col].unique()

        for label in labels:
            indices = df[df[groupby_col] == label].index.tolist()

            if len(indices) > 1:
                intra_group_similarities_values = []
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        intra_group_similarities_values.append(similarity_matrix[indices[i], indices[j]])
                intra_group_similarities[label] = np.mean(intra_group_similarities_values)
            
            else:
                intra_group_similarities[label] = 1.0
            
            for other_label in labels:
                if other_label != label:
                    other_indices = df[df[groupby_col] == other_label].index.tolist()
                    inter_group_similarities_values = []
                    for i in indices:
                        for j in other_indices:
                            inter_group_similarities_values.append(similarity_matrix[i,j])
                    
                    inter_group_similarities[(label, other_label)] = np.mean(inter_group_similarities_values)
        
        return intra_group_similarities, inter_group_similarities

    def calculate_cluster_metrics(self,embeddings, labels_list):
        labels_encoded = pd.factorize(labels_list)[0]
        
        silhouette = silhouette_score(embeddings, labels_encoded)
        
        davies_bouldin = davies_bouldin_score(embeddings, labels_encoded)
        
        return silhouette, davies_bouldin


    def get_questions_with_categories(self):
        with self.driver.session() as session:
            result = session.run("""
            MATCH (q:QnA)-[:BELONGS_TO]->(subcat:Subcategory)-[:CATEGORIZED_UNDER]->(cat:Category)
            RETURN q.identifier as q_id, q.question as question, subcat.name as subcategory, cat.name as category
            """)
            return pd.DataFrame([dict(record) for record in result])
    
    def suggest_adjustment(self,intra_similarity, inter_similarity, threshold_merge=0.8, threshold_split=0.4):
        merge_suggestions = []
        split_suggestions = []
        
        for (cat1, cat2), sim in inter_similarity.items():
            if sim > threshold_merge:
                merge_suggestions.append((cat1, cat2, sim))
        

        for cat, sim in intra_similarity.items():
            if sim < threshold_split:
                split_suggestions.append((cat, sim))
        
        return merge_suggestions, split_suggestions
    

hierarchy = Hierarchy()

st.subheader("Taxonomy Hierarchy")

chart_type = st.radio(
    "Select Visualization Type:",
    ["Sunburst", "Treemap"],
    horizontal=True
)
st.info("Click on the parent category to expand the subcategories.")
hierarchy.display_hierarchy(chart_type)


st.markdown("---")

st.subheader("Adjustment Analysis")

if st.button("Analyze"):
    with st.spinner("Calculating Similarities..."):
        hierarchy_analysis = HierarchyAnalysis()

        df = hierarchy_analysis.get_questions_with_categories()
        question_embeddings = hierarchy_analysis.model.encode(df['question'].tolist())
        similarity_matrix = cosine_similarity(question_embeddings)

        intra_similarity, inter_similarity = hierarchy_analysis.calculate_similarities(df, similarity_matrix, 'category')

        silhouette, db_index = hierarchy_analysis.calculate_cluster_metrics(question_embeddings, df['category'])

        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding_2d = reducer.fit_transform(question_embeddings)

        plot_df = pd.DataFrame({
            'TSNE1': embedding_2d[:, 0],
            'TSNE2': embedding_2d[:, 1],
            'category': df['category']
        })


        fig = px.scatter(
            plot_df, 
            x='TSNE1', 
            y='TSNE2',
            color='category',
            hover_data=['category'],
            title=f'TSNE Projection of Questions by Category<br>Silhouette: {silhouette:.3f}, Davies-Bouldin: {db_index:.3f}',
            opacity=0.7
        )

        # Customize layout
        fig.update_layout(
            legend_title_text='Categories',
            width=900,
            height=700,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        if silhouette > 0.8:
            analysis_message_silhouette = f"Silhouette Score: {silhouette:.3f}. The categorization of Q&A is very good."
        elif silhouette > 0.5:
            analysis_message_silhouette = f"Silhouette Score: {silhouette:.3f}. The categorization of Q&A is fair."
        elif silhouette < 0:
            analysis_message_silhouette = f"Silhouette Score: {silhouette}. Q&A pairs are assigned to wrong categories, as a different category is more similar."
        else:
            analysis_message_silhouette = f"Silhouette Score: {silhouette:.3f}. Categories seem to overlap, it may be good to split categories."
        

        st.info(f"{analysis_message_silhouette}")

        if db_index < 0.5:
            analysis_message_db = f"Davies-Bouldin Index: {db_index:.3f}. The categorization of Q&A is very good."
        elif db_index < 1:
            analysis_message_db = f"Davies-Bouldin Index: {db_index:.3f}. The categorization of Q&A is fair."
        else:
            analysis_message_db = f"Davies-Bouldin Index: {db_index:.3f}. Categorization of Q&A is not good."

        st.info(f"{analysis_message_db}")


        merge_suggestions, split_suggestions = hierarchy_analysis.suggest_adjustment(intra_similarity, inter_similarity)

        if len(merge_suggestions) > 0:
            st.subheader("Merge Suggestions")
            merge_df = pd.DataFrame(merge_suggestions, columns=['Category 1', 'Category 2', 'Similarity'])
            st.dataframe(merge_df)
        
        if len(split_suggestions) > 0:
            st.subheader("Split Suggestions")
            split_df = pd.DataFrame(split_suggestions, columns=['Category', 'Similarity'])
            st.dataframe(split_df)
