import streamlit as st
from knowledge_graph import KnowledgeGraph
from upload import UploadFile
import pandas as pd
import json
import datetime
from dateutil import parser


st.set_page_config(page_title="Home", layout="wide")

@st.cache_data
def get_taxonomy():
    return json.load(open('new_generated_data/taxonomy_hierarchy.json'))
taxonomy_hierarchy = get_taxonomy()

st.title("Q&A Knowledge Library")

@st.cache_data
def get_qna_data():
    from neo4j import GraphDatabase
    import os
    from dotenv import load_dotenv

    load_dotenv()

    neo4j_url = os.getenv('NEO4J_URL')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))

    with driver.session() as session:
        # retrieve QnA nodes with their related categories, subcategories and compliance standards
        # query = """
        # MATCH (qna:QnA)-[:BELONGS_TO]->(subcat:Subcategory)-[:CATEGORIZED_UNDER]->(cat:Category),
        # OPTIONAL MATCH 
        #       (qna)-[:COMPLIES_WITH]->(std:ComplianceStandard), (qna)-[:MAPS_TO]->(c:Control)
        # RETURN qna.identifier as id, qna.question as question, qna.answer as answer, 
        #        cat.name as category, subcat.name as subcategory, collect(std.name) as compliance_standards,
        #        qna.client_tag as client_tag, qna.note as note, qna.updated_at as updated_at, collect(c.name) as controls
        # ORDER BY qna.updated_at DESC
        # """

        #retrieve QnA nodes with their related categories, subcategories and compliance standards
        query = """
        MATCH (qna:QnA)-[:BELONGS_TO]->(subcat:Subcategory)-[:CATEGORIZED_UNDER]->(cat:Category),
            (qna)-[:COMPLIES_WITH]->(std:ComplianceStandard)
        OPTIONAL MATCH (qna)-[:MAPS_TO]->(c:Control)
        WITH qna, cat, subcat, collect(std) as standards, collect(c) as controls
        RETURN qna.identifier as id, 
            qna.question as question, 
            qna.answer as answer, 
            cat.name as category, 
            subcat.name as subcategory, 
            [std IN standards | std.name] as compliance_standards,
            qna.client_tag as client_tag, 
            qna.note as note, 
            qna.updated_at as updated_at, 
            CASE WHEN size(controls) = 0 OR all(x IN controls WHERE x IS NULL) 
                    THEN ['none'] 
                    ELSE [c IN controls WHERE c IS NOT NULL | c.name] 
            END as controls
        ORDER BY qna.updated_at DESC
        """
        
        result = session.run(query)
        records = [record.data() for record in result]
        return records

all_data = get_qna_data()
filtered_data = all_data


search_query = st.text_input(label="Search", placeholder="Search for questions, answers, or categories...")

if search_query:
    search_query_lower = search_query.lower()
    filtered_data = [item for item in filtered_data if 
                     search_query_lower in item['question'].lower() or 
                     search_query_lower in item['answer'].lower() or 
                     search_query_lower in item['category'].lower() or 
                     search_query_lower in item['subcategory'].lower()]

items_per_page = 10
if 'page' not in st.session_state:
    st.session_state.page = 0

total_pages = len(filtered_data) // items_per_page + (1 if len(all_data) % items_per_page > 0 else 0)

# Filter options
col1, col2, col3 = st.columns(3)

with col1:
    categories = list(taxonomy_hierarchy.keys())
    categories.insert(0, "Select Category")
    selected_category = st.selectbox("Category", categories)

with col2:
    subcategories = list(taxonomy_hierarchy[selected_category] if selected_category != "Select Category" else [])
    subcategories.insert(0, "Select Subcategory")
    selected_subcategory = st.selectbox("Subcategory", subcategories)

with col3:
    client_tags = list(set(item["client_tag"] for item in all_data))
    client_tags.insert(0, "Select Client")
    selected_client = st.selectbox("Client", client_tags)

add_1, add_2, _ = st.columns([0.5,0.5,4])

with add_1:
    @st.dialog("Add New Q&A")
    def add_qna():
        with st.form("add_qna"):
            question = st.text_area("Question", height=70)
            answer = st.text_area("Answer", height=70)
            note = st.text_input("Note/Comment", placeholder="Optional note/comment", value='')
            client = st.text_input("Client", placeholder="Client name to be used as a tag",value='not specified')

            with st.expander("Categorize yourself?"):
                cols = st.columns(2)
                with cols[0]:
                    add_categories = list(taxonomy_hierarchy.keys())
                    add_categories.insert(0, "Select Category")
                    category = st.selectbox(
                        "Category",
                        add_categories,
                    )
                
                with cols[1]:
                    add_subcategories = list(taxonomy_hierarchy[selected_category] if selected_category != "Select Category" else [])
                    add_subcategories.insert(0, "Select Subcategory")
                    subcategory = st.selectbox(
                        "Subcategory",
                        add_subcategories,
                    )          
            
            if st.form_submit_button("Submit Q&A", use_container_width=True, type='primary'):
                uploader = UploadFile()
                uploader.add_qna_pair(question, answer, note, client, category, subcategory)
                st.cache_data.clear()
                st.rerun()
    
    if st.button("Add New Q&A", key="add_new_qna"):
        add_qna()

with add_2:
    @st.dialog("Upload File")
    def file_upload():
        with st.form("file_upload"):
            client = st.text_input("Client", placeholder="Client name to be used as a tag")
            uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"], label_visibility='hidden')
            
            if st.form_submit_button("Submit"):

                with st.spinner('Uploading File...'):
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.client = client
                    
                    if st.session_state.uploaded_file is not None:
                        file = st.session_state.uploaded_file
                        uploader = UploadFile()

                        if file.name.endswith('.xlsx'):
                            df = pd.read_excel(file)
                            uploader.handle_file_qna_pair(df, st.session_state.client)
                        elif file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                            uploader.handle_file_qna_pair(df, st.session_state.client)
                        else:
                            st.error("Invalid file format. Please upload a CSV or Excel file.")

                        st.cache_data.clear()
                        st.rerun()

    if 'uploaded_file' in st.session_state:
        if 'client' in st.session_state:
            del st.session_state.client
        del st.session_state.uploaded_file
    
    if st.button("Upload Q&A"):
        file_upload()



if selected_category != "Select Category":
    filtered_data = [item for item in filtered_data if item["category"] == selected_category]

if selected_subcategory != "Select Subcategory":
    filtered_data = [item for item in filtered_data if item["subcategory"] == selected_subcategory]

if selected_client != "Select Client":
    filtered_data = [item for item in filtered_data if item["client_tag"] == selected_client]


total_pages = len(filtered_data) // items_per_page + (1 if len(filtered_data) % items_per_page > 0 else 0)

if st.session_state.page >= total_pages:
    st.session_state.page = max(0, total_pages - 1)


start_idx = st.session_state.page * items_per_page
end_idx = min(start_idx + items_per_page, len(filtered_data))

st.markdown("---")
if filtered_data:
    st.markdown(f"##### Showing {start_idx + 1}-{end_idx} of {len(filtered_data)} items")
    
    for i, item in enumerate(filtered_data[start_idx:end_idx]):
        with st.expander(f"{start_idx + i + 1} . {item['question']}"):
            cols = st.columns([3, 1, 1, 1])
            with cols[0]:
                st.write("**Answer:**")
                st.write(item['answer'])
            with cols[1]:
                st.write("**Category:**")
                st.write(item['category'])
            with cols[2]:
                st.write("**Subcategory:**")
                st.write(item['subcategory'])
            
            with cols[3]:
                st.write("**Compliance Standard:**")
                compliance_standards = item['compliance_standards']
                compliance_standards = (', ').join(compliance_standards) if len(compliance_standards) > 1 else compliance_standards[0]
                st.write(compliance_standards)
            
            def format_time_ago(timestamp_str):
                timestamp = parser.parse(timestamp_str)
                
                now = datetime.datetime.now(datetime.timezone.utc)

                diff = now - timestamp
                days_diff = diff.days

                if days_diff == 0:
                    return "today"
                elif days_diff == 1:
                    return "1 day ago"
                elif days_diff < 7:
                    return f"{days_diff} days ago"
                elif days_diff < 14:
                    return "1 week ago"
                else:
                    weeks = days_diff // 7
                    return f"{weeks} weeks ago"
                
            cols = st.columns([1, 1])
            with cols[0]:
                st.write(f"*Client:* {item['client_tag']}")
            with cols[1]:
                controls = item['controls']
                if len(controls) > 1:
                    st.write(f"**Related Policies**: {', '.join(controls)}")
                else:
                    st.write(f"**Related Policy**: {controls[0]}")

            st.write("*last updated:*", f'*{format_time_ago(str(item['updated_at']))}*')

            cols = st.columns([2, 2, 6])
            with cols[0]:
                if st.button(f"Edit Category Assignment", key=f"browse_edit_{item['id']}"):
                    st.session_state.editing_id = item['id']
                    st.session_state.editing_question = item['question']
                    st.session_state.editing_answer = item['answer']
                    st.session_state.editing_category = item['category']
                    st.session_state.editing_subcategory = item['subcategory']

                    @st.dialog("Edit Category Assignment", width=600)
                    def edit_qna():
                        with st.form("edit_qna"):

                            edited_question = st.text_area("Question", value=st.session_state.editing_question, height=70)
                            edited_answer = st.text_area("Answer", value=st.session_state.editing_answer, height=70)

                            # edited_question = st.success(f"Question: {st.session_state.editing_question}")
                            # edited_answer = st.info(f"Answer: {st.session_state.editing_answer}")
                            
                            all_subcategories = []
                            for category in taxonomy_hierarchy:
                                subcategories = taxonomy_hierarchy[category]
                                subcategory_with_category = [f"{category} - {subcategory}" for subcategory in subcategories]
                                all_subcategories.extend(subcategory_with_category)
                            all_subcategories = sorted(list(set(all_subcategories)))
                            
                            edited_subcategory = st.selectbox(
                                "Category - Subcategory", 
                                all_subcategories, 
                                index=subcategories.index(st.session_state.editing_subcategory) if st.session_state.editing_subcategory in subcategories else 0
                            )
                            st.write("   ")

                            _,update_col,_ = st.columns([0.5, 1, 0.5])

                            with update_col:
                                if st.form_submit_button("Update", use_container_width=True, type='primary'):
                                    updated_properties = {}
                                    if edited_question != st.session_state.editing_question:
                                        updated_properties['question'] = edited_question
                                    if edited_answer != st.session_state.editing_answer:
                                        updated_properties['answer'] = edited_answer
                                    
                                    if edited_subcategory != st.session_state.editing_subcategory:
                                        edited_category, edited_subcategory = edited_subcategory.split(' - ')
                                        updated_properties['category'] = edited_category
                                        updated_properties['subcategory'] = edited_subcategory
                                    kg = KnowledgeGraph()
                                    print("updated properties: ",updated_properties)
                                    update = kg.update_qna(identifier=st.session_state.editing_id, updated_properties=updated_properties)

                                    if update:
                                        st.success("Updated successfully!")
                                        st.cache_data.clear()
                                    else:
                                        st.error("Failed to update the item.")
                                    
                                    for key in ['editing_id', 'editing_question', 'editing_answer', 'editing_category', 'editing_subcategory']:
                                        if key in st.session_state:
                                            del st.session_state[key]
                                    
                                    st.rerun()
                                    
                    edit_qna()
                
                with cols[1]:
                    if st.button(f"Delete Q&A", key=f"delete_{item['id']}",type='primary'):
                        st.session_state.deleting_id = item['id']
                        kg = KnowledgeGraph()
                        kg.delete_qna(identifier=st.session_state.deleting_id)
                        
                        st.success("Deleted successfully!")
                        st.cache_data.clear()
                        del st.session_state.deleting_id
                        st.rerun()

else:
    st.info("No items match the selected filters.")


# Pagination controls
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.markdown(f"<p style='text-align: left;'>Page {st.session_state.page + 1} of {max(1, total_pages)}</p>", unsafe_allow_html=True)

with col3:
    btn_col1, btn_col2 = st.columns([0.5, 0.5])
    with btn_col1:
        if st.button("Previous Page", disabled=st.session_state.page <= 0):
            st.session_state.page -= 1
            st.rerun()

    with btn_col2:
        if st.button("Next Page", disabled=st.session_state.page >= total_pages - 1):
            st.session_state.page += 1
            st.rerun()

    