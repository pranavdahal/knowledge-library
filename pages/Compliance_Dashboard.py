import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from knowledge_graph import KnowledgeGraph

st.set_page_config(page_title="Compliance Mapping Dashboard", layout="wide")

@st.cache_data
def load_data_from_graph():

    kg = KnowledgeGraph()
    qa_pairs = pd.DataFrame(kg.get_qa_details())
    
    compliance_standards_data = kg.get_compliance_standards()
    compliance_standards = {f"cs_{i}": {"name": cs["compliance_standard_name"], "controls": cs["controls"]} 
                 for i, cs in enumerate(compliance_standards_data)}
    
    controls_df = pd.DataFrame(kg.get_compliance_controls())

    controls_df['compliance_standard'] = controls_df['compliance_standard_name'].apply(
        lambda x: next((k for k, v in compliance_standards.items() if v['name'] == x), None)
    )
    
    mappings_df = pd.DataFrame(kg.get_mappings())

    mappings_df['compliance_standard'] = mappings_df['compliance_standard_name'].apply(
        lambda x: next((k for k, v in compliance_standards.items() if v['name'] == x), None)
    )
    
    return qa_pairs, controls_df, mappings_df, compliance_standards

qa_pairs, controls_df, mappings_df, compliance_standards = load_data_from_graph()


st.sidebar.title("Compliance Mapping")
selected_compliance_standard = st.sidebar.selectbox(
    "Select Compliance Standard",
    options=list(compliance_standards.keys()),
    format_func=lambda x: compliance_standards[x]['name']
)

st.session_state.coverage_threshold = st.sidebar.slider("Criteria for coverage",min_value=1,max_value=10,value=3,step=1)
st.sidebar.info("Adjust the slider to set the minimum number of mappings required for a control to be considered 'covered'.")

st.title(f"Compliance Standard Gap Analysis ({compliance_standards[selected_compliance_standard]['name']})")
st.markdown("---")

def generate_compliance_standard_stats(compliance_standard_id):
    compliance_standard_controls = controls_df[controls_df['compliance_standard'] == compliance_standard_id]
    total_controls = len(compliance_standard_controls)
    
    mapped_control_ids = mappings_df[mappings_df['compliance_standard'] == compliance_standard_id]['control_id'].unique()
    mapped_controls = len(mapped_control_ids)
    
    control_coverage = {}
    for control_id in compliance_standard_controls['id']:
        control_mappings = mappings_df[mappings_df['control_id'] == control_id]
        if len(control_mappings) == 0:
            status = 'missing'
        elif len(control_mappings) < st.session_state.coverage_threshold:
            status = 'partial'
        else:
            status = 'covered'
        
        if control_id not in control_coverage:
            control_coverage[control_id] = status
    
    covered = sum(1 for status in control_coverage.values() if status == 'covered')
    partial = sum(1 for status in control_coverage.values() if status == 'partial')
    missing = sum(1 for status in control_coverage.values() if status == 'missing')
    
    return {
        'total': total_controls,
        'mapped': mapped_controls,
        'covered': covered,
        'partial': partial,
        'missing': missing
    }

stats = generate_compliance_standard_stats(selected_compliance_standard)

st.subheader("Controls Coverage Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="", value='')

with col2:
    covered_pct = stats['covered']/stats['total']*100 if stats['total'] > 0 else 0
    fig2 = go.Figure(go.Pie(
        values=[covered_pct, 100-covered_pct],
        hole=0.7,
        marker_colors=['#196f3d', '#e6e6e6'],
        showlegend=False,
        hoverinfo='none',
        sort=False
    ))
    fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200, width=200)
    st.plotly_chart(fig2)

with col3:
    partial_pct = stats['partial']/stats['total']*100 if stats['total'] > 0 else 0
    fig3 = go.Figure(go.Pie(
        values=[partial_pct, 100-partial_pct],
        hole=0.7,
        marker_colors=['#ff9900', '#e6e6e6'],
        showlegend=False,
        hoverinfo='none'
    ))
    fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200, width=200)
    st.plotly_chart(fig3)

with col4:
    missing_pct = stats['missing']/stats['total']*100 if stats['total'] > 0 else 0
    fig4 = go.Figure(go.Pie(
        values=[missing_pct, 100-missing_pct],
        hole=0.7,
        marker_colors=['#dc3545', '#e6e6e6'],
        showlegend=False,
        hoverinfo='none'
    ))
    fig4.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200, width=200)
    st.plotly_chart(fig4)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Controls", value=stats['total'], border=True)
with col2:
    st.metric(label="Fully Covered", value=stats['covered'],border=True)
with col3:
    st.metric(label="Partially Covered", value=stats['partial'],border=True)
with col4:
    st.metric(label="Gaps (No Coverage)", value=stats['missing'],delta_color='off', border=True)


def generate_coverage_dataframe(compliance_standard_id):
    standard_controls = controls_df[controls_df['compliance_standard'] == compliance_standard_id]
    
    coverage_details = []
    
    for _, control in standard_controls.iterrows():
        control_id = control['id']
        control_name = control['name']
        domain = control.get('domain', 'Not specified')
        
        # Count mappings for this control
        control_mappings = mappings_df[mappings_df['control_id'] == control_id]
        mapping_count = len(control_mappings)
        
        # Determine coverage status
        if mapping_count == 0:
            status = f'No Coverage'
        elif mapping_count < st.session_state.coverage_threshold:
            status = f'Partial'
        else:
            status = f'Covered'
        
        coverage_details.append({
            'control_id': control_id,
            'control_name': control_name,
            'domain': domain,
            'coverage_status': status
        })
    

    coverage_df = pd.DataFrame(coverage_details)
    
    return coverage_df

with st.expander("Expand Controls Coverage Details"):
    coverage_df = generate_coverage_dataframe(selected_compliance_standard)

    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "Filter by Coverage Status",
            options=["Covered", "Partial", "No Coverage"],
            default=[]
        )
    with col2:
        if 'domain' in coverage_df.columns and len(coverage_df['domain'].unique()) > 1:
            domain_filter = st.multiselect(
                "Filter by Domain",
                options=sorted(coverage_df['domain'].unique()),
                default=[]
            )
        else:
            domain_filter = sorted(coverage_df['domain'].unique()) if 'domain' in coverage_df.columns else []

    filtered_df = coverage_df
    if status_filter:
        filtered_df = filtered_df[filtered_df['coverage_status'].isin(status_filter)]
    if domain_filter:
        filtered_df = filtered_df[filtered_df['domain'].isin(domain_filter)]

    st.dataframe(filtered_df,width=1500)



st.markdown('---')
st.header("Gap Analysis")
gaps_df = coverage_df[coverage_df['coverage_status'].isin(['No Coverage', 'Partial'])]

st.subheader("Gaps by Domain and Controls")
if len(gaps_df) > 0 and 'domain' in gaps_df.columns:
    labels = []
    parents = []
    values = []
    colors = []
    hover_texts = []

    for domain in gaps_df['domain'].unique():
        domain_controls = gaps_df[gaps_df['domain'] == domain]
        
        labels.append(domain)
        parents.append("")
        values.append(len(domain_controls))
        colors.append("#c9cde7")  

        for _, control in domain_controls.iterrows():
            labels.append(f"{control['control_id']}")
            parents.append(domain)
            values.append(1)
            
            if control['coverage_status'] == 'Partial':
                color = "#f0ed2f"  
            else: 
                color = "#eb4040"
                
            colors.append(color)
    

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>Number of Controls: %{value}<extra></extra>'
    ))
    
    fig.update_layout(
        title="🟨 Controls with Partial Coverage  🟥 Controls with No Coverage",
        margin=dict(t=30, l=10, r=10, b=10),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
elif len(gaps_df) == 0:
    st.success("All controls have sufficient coverage!")
else:
    st.info("Domain information is missing.")

# if len(gaps_df) > 0:
#     st.warning(f"{len(gaps_df)} controls have insufficient coverage")
    
#     # by domain and coverage status
#     if 'domain' in gaps_df.columns:
#         domain_gaps = gaps_df.groupby(['domain', 'coverage_status']).size().reset_index(name='count')
        
#         fig = px.bar(
#             domain_gaps, 
#             x='domain', 
#             y='count',
#             color='coverage_status',
#             color_discrete_map={'Missing': '#dc3545', 'Partial': '#ff9900'},
#             title="Coverage Gaps by Domain",
#             labels={'domain': 'Domain', 'count': 'Number of Controls', 'coverage_status': 'Coverage Status'}
#         )
#         st.plotly_chart(fig)
    
st.markdown('---')
st.subheader("Mapping Density Analysis")

# qna count for each control in the selected standard
controls_in_standard = controls_df[controls_df['compliance_standard'] == selected_compliance_standard]
control_mapping_counts = []

for _, control in controls_in_standard.iterrows():
    control_id = control['id']
    control_name = control['name']
    mapping_count = len(mappings_df[mappings_df['control_id'] == control_id])
    
    control_mapping_counts.append({
        'control_id': control_id,
        'control_name': control_name,
        'mapping_count': mapping_count,
        'domain': control.get('domain', 'Not specified')
    })

density_df = pd.DataFrame(control_mapping_counts)
density_df = density_df.sort_values('mapping_count', ascending=False)

if not density_df.empty:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        top_n = min(10, len(density_df))  
        fig = px.bar(
            density_df.head(top_n), 
            x='control_id', 
            y='mapping_count',
            hover_data=['control_name', 'domain'],
            labels={'control_id': 'Control ID', 'mapping_count': 'Number of Mappings'},
            title=f"Top {top_n} Controls by Mapping Count",
            color='mapping_count',
            color_continuous_scale='Viridis'
        )
        fig.update_xaxes(
            type='category',
            tickmode='array',
            tickvals=density_df.head(top_n)['control_id'].tolist(),
            ticktext=density_df.head(top_n)['control_id'].tolist()
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
    
    with col2:
        st.metric("Average Mappings per Control", 
                 round(density_df['mapping_count'].mean(), 1))
        
        st.metric("Most Mapped Control", 
                 f"{density_df.iloc[0]['control_name']} ({density_df.iloc[0]['mapping_count']})")
        
        st.write("Mapping Distribution:")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("No Mappings", 
                    len(density_df[density_df['mapping_count'] == 0]))
        with col_b:
            st.metric(f"{st.session_state.coverage_threshold}+ Mappings", 
                    len(density_df[density_df['mapping_count'] >= st.session_state.coverage_threshold]))
else:
    st.info("No controls found in the selected compliance standard.")


st.sidebar.markdown("---")
if st.sidebar.button("Export to Excel", type='primary', use_container_width=True):
    import io
    from io import BytesIO

    @st.dialog("Export Excel File")
    def export_excel_dialog():
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            gaps_df.to_excel(writer, index=False, sheet_name='Gaps Analysis')
        
        buffer.seek(0)
        
        st.write("Your file is ready!")
        
        st.download_button(
            label="Download", 
            data=buffer,
            file_name="gaps_analysis.xlsx", 
            mime="application/vnd.ms-excel", 
            type='primary'
        )
    export_excel_dialog()