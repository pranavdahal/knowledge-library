from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

class KnowledgeGraph:
    def __init__(self):
        self.neo4j_url = os.getenv('NEO4J_URL')
        self.neo4j_username = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password))

        with self.driver.session() as session:
            session.run('CREATE CONSTRAINT qna_id_unique IF NOT EXISTS FOR (q:QnA) REQUIRE q.identifier IS UNIQUE')
            session.run('CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE')
            session.run('CREATE CONSTRAINT subcategory_name_unique IF NOT EXISTS FOR (s:Subcategory) REQUIRE s.name IS UNIQUE')
            session.run('CREATE CONSTRAINT compliance_standard_name_unique IF NOT EXISTS FOR (cs:ComplianceStandard) REQUIRE cs.name IS UNIQUE')
    
    # cypher query for compliance standards
    def create_standard(self, name, version):
        with self.driver.session() as session:
            query = """
            MERGE (s:ComplianceStandard {name: $name, version: $version})
            RETURN s.name + ' ' + s.version as standard
            """
            result = session.run(query, name=name, version=version)
            return result.single()[0]
        
    def create_domain(self, domain_id, domain_name, standard_name, standard_version):
        with self.driver.session() as session:
            query = """
            MATCH (s:ComplianceStandard {name: $standard_name, version: $standard_version})
            MERGE (d:Domain {id: $domain_id, name: $domain_name})
            MERGE (d)-[:PART_OF]->(s)
            RETURN d.id + ': ' + d.name as domain
            """
            result = session.run(
                query, 
                domain_id=domain_id, 
                domain_name=domain_name, 
                standard_name=standard_name, 
                standard_version=standard_version
            )
            return result.single()[0]
    
    def create_control(self, control_id, title, description, domain_id, domain_name):
        with self.driver.session() as session:
            query = """
            MATCH (d:Domain {id: $domain_id, name: $domain_name})
            MERGE (c:Control {id: $control_id, name: $title, description: $description})
            MERGE (c)-[:BELONGS_TO]->(d)
            RETURN c.id + ': ' + c.name as control
            """
            result = session.run(
                query, 
                control_id=control_id, 
                title=title, 
                description=description, 
                domain_id=domain_id,
                domain_name=domain_name
            )
            return result.single()[0]

    def create_control_relationship(self, control_id, related_control_id):
        """Create a relationship between two control nodes"""
        with self.driver.session() as session:
            query = """
            MATCH (c1:Control {id: $control_id})
            MATCH (c2:Control {id: $related_control_id})
            MERGE (c1)-[r:RELATED_TO]->(c2)
            RETURN c1.id + ' -> ' + c2.id as relationship
            """
            result = session.run(query, control_id=control_id, related_control_id=related_control_id)
            record = result.single()
            return record[0] if record else None

    
    # cypher query for qna
    def add_qna_data_to_graph(self, df:pd.DataFrame, client_name:str):

        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        if 'notes/comments' in df.columns:
            df['notes/comments'].fillna('', inplace=True)
        
        if 'answer' in df.columns:
            df['answer'].fillna('', inplace=True)

        if 'question_text' in df.columns:
            df['question_text'].fillna('', inplace=True)
        elif 'question' in df.columns:
            df['question'].fillna('', inplace=True)

        query = """
            MERGE (qna:QnA {identifier:$identifier, question: $question, answer: $answer, client_tag: $client_tag, note:$note, created_at: datetime(), updated_at: datetime()})
            MERGE (cat: Category {name: $category})
            MERGE (subcat: Subcategory {name: $subcategory})
            MERGE (std: ComplianceStandard {name: $compliance_standard})
            MERGE (qna)-[:BELONGS_TO]->(subcat)
            MERGE (subcat)-[:CATEGORIZED_UNDER]->(cat)
            MERGE (qna)-[:COMPLIES_WITH]->(std)
            """

        compliance_standard = 'none'

        with self.driver.session() as session:
            for index, row in df.iterrows():
                id = uuid.uuid4()

                question = row['question_text'] if 'question_text' in df.columns else row['question']
                if question is None:
                    question = ''

                answer = row['answer'] if 'answer' in df.columns else ''
                if answer is None:
                    answer = ''

                note = row['notes/comments'] if 'notes/comments' in df.columns else ''

                category = row['category'] if 'category' in df.columns else 'uncategorized'
                subcategory = row['subcategory'] if 'subcategory' in df.columns else 'uncategorized'
                
                session.run(query, identifier = str(id), question = question, answer = answer, client_tag = client_name, note=note,category = category, subcategory = subcategory, compliance_standard = compliance_standard)

    def update_qna(self, identifier, updated_properties):
        updated_properties['updated_at'] = datetime.now()

        question = updated_properties.pop('question', None)
        answer = updated_properties.pop('answer', None)
        category = updated_properties.pop('category', None)
        subcategory = updated_properties.pop('subcategory', None)
        compliance_standard = updated_properties.pop('compliance_standard', None)
        
    
        with self.driver.session() as session:
            if question or answer:
                qna_query = """
                MATCH (qna:QnA {identifier: $identifier})
                SET qna.question = $question
                SET qna.answer = $answer
                RETURN qna
                """
                session.run(qna_query, identifier=identifier, question=question, answer=answer)
            
            # update relationships
            if subcategory:
                subcategory_query = """
                MATCH (qna:QnA {identifier: $identifier})
                OPTIONAL MATCH (qna)-[r1:BELONGS_TO]->(:Subcategory)
                DELETE r1
                WITH qna
                MERGE (subcat:Subcategory {name: $subcategory})
                MERGE (qna)-[:BELONGS_TO]->(subcat)
                RETURN qna, subcat
                """
                session.run(subcategory_query, identifier=identifier, subcategory=subcategory)
                
            if category and subcategory:
                category_query = """
                MATCH (subcat:Subcategory {name: $subcategory})
                OPTIONAL MATCH (subcat)-[r2:CATEGORIZED_UNDER]->(:Category)
                DELETE r2
                WITH subcat
                MERGE (cat:Category {name: $category})
                MERGE (subcat)-[:CATEGORIZED_UNDER]->(cat)
                RETURN subcat, cat
                """
                session.run(category_query, subcategory=subcategory, category=category)
                
            if compliance_standard:
                compliance_query = """
                MATCH (qna:QnA {identifier: $identifier})
                OPTIONAL MATCH (qna)-[r3:COMPLIES_WITH]->(:ComplianceStandard)
                DELETE r3
                WITH qna
                MERGE (std:ComplianceStandard {name: $compliance_standard})
                MERGE (qna)-[:COMPLIES_WITH]->(std)
                RETURN qna, std
                """
                session.run(compliance_query, identifier=identifier, compliance_standard=compliance_standard)
            
            return True
    
    def delete_qna(self, identifier):
        with self.driver.session() as session:
            query = """
            MATCH (q:QnA {identifier: $identifier})
            DETACH DELETE q
            """
            session.run(query, identifier=identifier)
            return True
        
    def add_qna_pair(self, question, answer, client, category, subcategory, note='', compliance_standard="none"):            
        
        query = """
            MERGE (qna:QnA {identifier:$identifier, question: $question, answer: $answer, client_tag: $client_tag, note:$note, created_at: datetime(), updated_at: datetime()})
            MERGE (cat: Category {name: $category})
            MERGE (subcat: Subcategory {name: $subcategory})
            MERGE (std: ComplianceStandard {name: $compliance_standard})
            MERGE (qna)-[:BELONGS_TO]->(subcat)
            MERGE (subcat)-[:CATEGORIZED_UNDER]->(cat)
            MERGE (qna)-[:COMPLIES_WITH]->(std)
            """
        id = uuid.uuid4()

        with self.driver.session() as session:
            session.run(query,identifier = str(id), question = question, answer = answer, client_tag = client, note=note, category = category, subcategory = subcategory, compliance_standard = compliance_standard)
    
    def update_qna_compliance_mapping(self,qna_id, control_id, control_name,similarity_score):
        with self.driver.session() as session:
            query = """
            MATCH (q:QnA {identifier: $qna_id})-[r:COMPLIES_WITH]->(s:ComplianceStandard)
            DELETE r
            WITH q
            MATCH (c:Control {id: $control_id, name: $control_name})
            MATCH (s:ComplianceStandard)<-[:PART_OF]-(d:Domain)<-[:BELONGS_TO]-(c)
            MERGE (q)-[:COMPLIES_WITH]->(s)
            MERGE (q)-[:MAPS_TO {similarity_score: $score}]->(c)
            """
            session.run(query, qna_id=qna_id, control_id=control_id, control_name=control_name,score=similarity_score)
    
    def get_qna_pairs_client(self,client_name):
        with self.driver.session() as session:
            query = """
            MATCH (q:QnA {client_tag: $client_name})-[:COMPLIES_WITH]->(s:ComplianceStandard {name: 'none'})
            RETURN q.identifier AS id, q.question AS question, q.answer AS answer
            """
            return session.run(query, client_name=client_name).data()
            
    
    def get_qna_pairs(self):
        with self.driver.session() as session:
            query = """
            MATCH (q:QnA)-[:COMPLIES_WITH]->(s:ComplianceStandard {name: 'none'})
            RETURN q.identifier AS id, q.question AS question, q.answer AS answer
            """
            return session.run(query).data()
    
    def get_qa_details(self):
        with self.driver.session() as session:
            query = """
            MATCH (q:QnA)-[:BELONGS_TO]->(sub:Subcategory)-[:CATEGORIZED_UNDER]->(cat:Category)
            RETURN q.identifier as id, q.question as question, q.answer as answer, 
                cat.name as category, sub.name as subcategory
            """
            return session.run(query).data()
    
    def get_compliance_standards(self):
        with self.driver.session() as session:
            query = """
            MATCH (s:ComplianceStandard)<-[:PART_OF]-(d:Domain)<-[:BELONGS_TO]-(c:Control)
            RETURN s.name as compliance_standard_name, count(distinct c) as controls
            """
            return session.run(query).data()

    def get_compliance_controls(self):
        with self.driver.session() as session:
            query = """
            MATCH (s:ComplianceStandard)<-[:PART_OF]-(d:Domain)<-[:BELONGS_TO]-(c:Control)
            RETURN c.id as id, s.name as compliance_standard_name, 
                d.name as domain, c.name as name, 
                c.description as description
            """
            return session.run(query).data()
    
    def get_all_controls(self):
        with self.driver.session() as session:
            query = """
            MATCH (c:Control)-[:BELONGS_TO]->(d:Domain)-[:PART_OF]->(s:ComplianceStandard)
            WHERE s.name <> 'none'
            RETURN c.id AS id, c.name AS control_name, c.description AS description, 
                s.name AS standard_name, d.name AS domain_name
            """
            return session.run(query).data()
    
    def get_mappings(self):
        with self.driver.session() as session:
            query = """
            MATCH (q:QnA)-[m:MAPS_TO]->(c:Control)-[:BELONGS_TO]->()-[:PART_OF]->(s:ComplianceStandard)
            RETURN q.identifier as qa_id, c.id as control_id, s.name as compliance_standard_name
            """
            return session.run(query).data()