from neo4j import GraphDatabase
import json
import os

from dotenv import load_dotenv
load_dotenv()


class ComplianceImporter:
    def __init__(self, url=None, username=None, password=None):
        self.url = url
        self.username = username
        self.password = password

        if self.url:
            if self.username and self.password:
                self.driver = GraphDatabase.driver(url, auth=(username, password))
            else:
                return "Username and password are required"
        else:
            self.url = os.getenv('NEO4J_URL')
            self.username = os.getenv('NEO4J_USERNAME')
            self.password = os.getenv('NEO4J_PASSWORD')
            self.driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))

    def close(self):
        self.driver.close()

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
    
    def create_control(self, control_id, title, description, domain_id):
        with self.driver.session() as session:
            query = """
            MATCH (d:Domain {id: $domain_id})
            MERGE (c:Control {id: $control_id, title: $title, description: $description})
            MERGE (c)-[:BELONGS_TO]->(d)
            RETURN c.id + ': ' + c.title as control
            """
            result = session.run(
                query, 
                control_id=control_id, 
                title=title, 
                description=description, 
                domain_id=domain_id
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

    def import_compliance_standard(self, json_file_path):
        try:
            with open(json_file_path, 'r') as file:
                compliance_data = json.load(file)
            
            standard_name = compliance_data.get('standard')
            standard_version = compliance_data.get('version')
            
            if not standard_name:
                return False
            
            self.create_standard(standard_name, standard_version)
            
            domains = compliance_data.get('domains', [])
            for domain in domains:
                domain_id = domain.get('domain_id')
                domain_name = domain.get('domain_name')
                
                if not domain_id or not domain_name:
                    continue
                    
                self.create_domain(domain_id, domain_name, standard_name, standard_version)
                
                controls = domain.get('controls', [])
                for control in controls:
                    control_id = control.get('control_id')
                    title = control.get('title')
                    description = control.get('description')
                    
                    if not control_id or not title:
                        continue
                    
                    self.create_control(
                        control_id, 
                        title, 
                        description,
                        domain_id
                    )
            
            for domain in domains:
                controls = domain.get('controls', [])
                for control in controls:
                    control_id = control.get('control_id')
                    related_controls = control.get('related_controls', [])
                    
                    for related_id in related_controls:
                        self.create_control_relationship(control_id, related_id)
            
            return True
            
        except Exception as e:
            return False

    def import_all_standards(self, directory_path):
        count = 0
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                if self.import_compliance_standard(file_path):
                    count += 1
        return count

if __name__ == "__main__":    
    importer = ComplianceImporter()
    
    try:
        importer.import_all_standards("new_generated_data/compliance_standards")
        
    finally:
        importer.close()