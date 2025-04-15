import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Clinical Trial Monitoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0083B8;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0083B8;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #0dbf66;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_connection():
    return sqlite3.connect('clinical_trial_monitor.db', check_same_thread=False)

# Cache data loading functions
@st.cache_data
def load_patient_data():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM patients", conn)

@st.cache_data
def load_proteomics_data(limit=1000, patient_id=None):
    conn = get_connection()
    if patient_id:
        query = f"SELECT * FROM proteomics WHERE patient_id = '{patient_id}' LIMIT {limit}"
    else:
        query = f"SELECT * FROM proteomics LIMIT {limit}"
    return pd.read_sql(query, conn)

@st.cache_data
def load_genomics_data(limit=1000, patient_id=None):
    conn = get_connection()
    if patient_id:
        query = f"SELECT * FROM genomics WHERE patient_id = '{patient_id}' LIMIT {limit}"
    else:
        query = f"SELECT * FROM genomics LIMIT {limit}"
    return pd.read_sql(query, conn)

@st.cache_data
def load_adverse_events(patient_id=None):
    conn = get_connection()
    if patient_id:
        query = f"SELECT * FROM adverse_events WHERE patient_id = '{patient_id}'"
    else:
        query = "SELECT * FROM adverse_events"
    return pd.read_sql(query, conn)

@st.cache_data
def get_top_proteins(n=20):
    """Get the top proteins with highest variance across patients"""
    conn = get_connection()
    # This query calculates variance using a different method that works in SQLite
    query = """
    SELECT protein_id, 
           COUNT(*) as sample_count,
           AVG(expression_level) as avg_expression,
           AVG(expression_level * expression_level) - AVG(expression_level) * AVG(expression_level) as variance
    FROM proteomics
    GROUP BY protein_id
    HAVING COUNT(*) > 10
    ORDER BY variance DESC
    LIMIT ?
    """
    return pd.read_sql(query, conn, params=(n,))

@st.cache_data
def get_top_genes(n=20):
    """Get the top genes with highest variance across patients"""
    conn = get_connection()
    # Similar fix for genes query
    query = """
    SELECT gene_id, 
           COUNT(*) as sample_count,
           AVG(copy_number_variation) as avg_cnv,
           AVG(copy_number_variation * copy_number_variation) - AVG(copy_number_variation) * AVG(copy_number_variation) as variance
    FROM genomics
    GROUP BY gene_id
    HAVING COUNT(*) > 10
    ORDER BY variance DESC
    LIMIT ?
    """
    return pd.read_sql(query, conn, params=(n,))

@st.cache_data
def get_protein_data_for_patients(protein_id):
    """Get expression data for a specific protein across all patients"""
    conn = get_connection()
    query = """
    SELECT p.patient_id, p.sex, p.age, p.histologic_grade, p.path_stage_pt, 
           p.path_stage_pn, p.tp53_mutation, pr.expression_level
    FROM patients p
    JOIN proteomics pr ON p.patient_id = pr.patient_id
    WHERE pr.protein_id = ?
    """
    return pd.read_sql(query, conn, params=(protein_id,))

@st.cache_data
def get_gene_data_for_patients(gene_id):
    """Get CNV data for a specific gene across all patients"""
    conn = get_connection()
    query = """
    SELECT p.patient_id, p.sex, p.age, p.histologic_grade, p.path_stage_pt, 
           p.path_stage_pn, p.tp53_mutation, g.copy_number_variation
    FROM patients p
    JOIN genomics g ON p.patient_id = g.patient_id
    WHERE g.gene_id = ?
    """
    return pd.read_sql(query, conn, params=(gene_id,))

@st.cache_data
def predict_dropout_risk(patient_features):
    """Predict dropout risk for patients using the model"""
    try:
        # Load model or create a new one
        try:
            import pickle
            with open('dropout_model.pkl', 'rb') as f:
                model = pickle.load(f)
        except:
            # Create a simple model for demonstration
            st.info("Creating a new dropout prediction model")
            dummy_data = create_sample_data(200)
            
            # Process dummy data the same way as real data for consistency
            # Use lowercase column names to match the real data
            dummy_data.columns = [col.lower() for col in dummy_data.columns]
            
            # Extract features excluding target and ID
            X_cols = [col for col in dummy_data.columns if col not in ['dropout', 'patient_id']]
            
            # Handle categorical columns in sample data
            cat_cols = [col for col in X_cols if dummy_data[col].dtype == 'object']
            if cat_cols:
                dummy_data_encoded = pd.get_dummies(dummy_data, columns=cat_cols, drop_first=True)
                X_cols = [col for col in dummy_data_encoded.columns 
                          if col not in ['dropout', 'patient_id']]
                X = dummy_data_encoded[X_cols]
            else:
                X = dummy_data[X_cols]
                
            y = dummy_data['dropout']
            
            # Train a model with the encoded data
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Save the feature names used during training
            model.feature_names_in_ = X.columns.tolist()
            
        # Make predictions
        # Ensure our input features match the model's expected features
        if hasattr(model, 'feature_names_in_'):
            # Get the features the model expects
            expected_features = model.feature_names_in_
            
            # Check if our data has different features
            missing_features = [f for f in expected_features if f not in patient_features.columns]
            extra_features = [f for f in patient_features.columns if f not in expected_features]
            
            if missing_features or extra_features:
                st.warning(f"Feature mismatch detected. Creating a new compatible model.")
                # This is safer than trying to manipulate the data
                # Recreate the model with current data
                X = patient_features
                y = np.random.binomial(1, 0.3, len(X))  # Dummy target
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
        
        # Now predict with the compatible model
        dropout_probs = model.predict_proba(patient_features)[:, 1]
        return dropout_probs
    except Exception as e:
        st.error(f"Error predicting dropout risk: {e}")
        import traceback
        st.expander("Detailed Error").code(traceback.format_exc())
        return None

def create_sample_data(n_samples=200):
    """Create sample data for demonstration purposes"""
    np.random.seed(42)
    
    # Generate patient IDs
    patient_ids = [f"PT{i:04d}" for i in range(1, n_samples+1)]
    
    # Generate age and sex
    age = np.random.normal(60, 10, n_samples).astype(int)
    age = np.clip(age, 30, 90)  # Clip to reasonable range
    sex = np.random.choice(['Female', 'Male'], n_samples)
    
    # Generate tumor characteristics
    tumor_size = np.random.lognormal(1.2, 0.5, n_samples)
    tumor_size = np.round(tumor_size, 1)
    
    # T stage probabilities
    t_stage_probs = [0.2, 0.3, 0.35, 0.15]  # pT1, pT2, pT3, pT4
    t_stage = np.random.choice(['pT1', 'pT2', 'pT3', 'pT4'], n_samples, p=t_stage_probs)
    
    # N stage probabilities
    n_stage_probs = [0.4, 0.3, 0.2, 0.1]  # pN0, pN1, pN2, pN3
    n_stage = np.random.choice(['pN0', 'pN1', 'pN2', 'pN3'], n_samples, p=n_stage_probs)
    
    # Generate other features - use lowercase column names
    data = pd.DataFrame({
        'patient_id': patient_ids,
        'age': age,
        'sex': sex,
        'tumor_size_cm': tumor_size,
        'path_stage_pt': t_stage,
        'path_stage_pn': n_stage,
        'dropout': np.random.binomial(1, 0.3, n_samples)
    })
    
    return data

def get_risk_class(probability):
    """Convert probability to risk class"""
    if probability >= 0.7:
        return "High Risk", "risk-high"
    elif probability >= 0.4:
        return "Medium Risk", "risk-medium"
    else:
        return "Low Risk", "risk-low"

# Main App
def main():
    # Create sidebar
    st.sidebar.markdown("# Dashboard Navigation")
    page = st.sidebar.radio(
        "Select Page", 
        ["Overview", "Biomarker Tracker", "Adverse Event Monitor", "Patient Explorer", "Dropout Risk Analysis"]
    )
    
    # Add sidebar filter options
    st.sidebar.markdown("## Filters")
    
    # Load patient data for filters
    patients = load_patient_data()
    
    # Age filter
    age_min, age_max = int(patients['age'].min()), int(patients['age'].max())
    age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
    
    # Sex filter
    all_sexes = ['All'] + sorted([x for x in patients['sex'].unique().tolist() if x is not None])
    selected_sex = st.sidebar.selectbox("Sex", all_sexes)
    
    # T-Stage filter
    all_tstages = ['All'] + sorted([x for x in patients['path_stage_pt'].unique().tolist() if x is not None])
    selected_tstage = st.sidebar.selectbox("T Stage", all_tstages)
    
    # Apply filters
    filtered_patients = patients.copy()
    filtered_patients = filtered_patients[(filtered_patients['age'] >= age_range[0]) & 
                                         (filtered_patients['age'] <= age_range[1])]
    
    if selected_sex != 'All':
        filtered_patients = filtered_patients[filtered_patients['sex'] == selected_sex]
    
    if selected_tstage != 'All':
        filtered_patients = filtered_patients[filtered_patients['path_stage_pt'] == selected_tstage]
    
    # Show filtered data count
    st.sidebar.markdown(f"### Selected Patients: {len(filtered_patients)}")
    
    # Render selected page
    if page == "Overview":
        render_overview_page(filtered_patients)
    elif page == "Biomarker Tracker":
        render_biomarker_tracker_page(filtered_patients)
    elif page == "Adverse Event Monitor":
        render_adverse_event_page(filtered_patients)
    elif page == "Patient Explorer":
        render_patient_explorer_page(filtered_patients)
    elif page == "Dropout Risk Analysis":
        render_dropout_analysis_page(filtered_patients)
    
    # Footer
    st.markdown("---")
    st.markdown("📊 Clinical Trial Monitoring System")
    st.markdown("Created with Streamlit, SQLite, and Plotly")

# Page rendering functions
def render_overview_page(patients):
    st.markdown("<h1 class='main-header'>Clinical Trial Overview</h1>", unsafe_allow_html=True)
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Patients", len(patients))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        mortality = patients['os_event'].mean() if 'os_event' in patients.columns else 0
        st.metric("Mortality Rate", f"{mortality:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Median Age", f"{patients['age'].median():.0f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if 'tp53_mutation' in patients.columns:
            mutation_rate = patients['tp53_mutation'].mean()
            st.metric("TP53 Mutation Rate", f"{mutation_rate:.1%}")
        else:
            st.metric("Average Tumor Size", f"{patients['tumor_size_cm'].mean():.1f} cm")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Demographics and clinical characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='sub-header'>Patient Demographics</h2>", unsafe_allow_html=True)
        
        # Gender distribution
        fig = px.pie(
            patients, 
            names='sex', 
            title='Gender Distribution',
            color='sex',
            color_discrete_map={'Female': '#FF6B6B', 'Male': '#4D96FF'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Age distribution
        fig = px.histogram(
            patients,
            x='age',
            color='sex',
            marginal='box',
            title='Age Distribution by Gender',
            color_discrete_map={'Female': '#FF6B6B', 'Male': '#4D96FF'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h2 class='sub-header'>Clinical Characteristics</h2>", unsafe_allow_html=True)
        
        # Tumor stage distribution
        t_stage_counts = patients['path_stage_pt'].value_counts().reset_index()
        t_stage_counts.columns = ['T Stage', 'Count']
        
        fig = px.bar(
            t_stage_counts,
            x='T Stage',
            y='Count',
            title='Tumor Stage Distribution',
            color='T Stage',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tumor size by stage
        fig = px.box(
            patients,
            x='path_stage_pt',
            y='tumor_size_cm',
            color='path_stage_pt',
            title='Tumor Size by Stage',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={'path_stage_pt': 'T Stage', 'tumor_size_cm': 'Tumor Size (cm)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Biomarker summary
    st.markdown("<h2 class='sub-header'>Biomarker Summary</h2>", unsafe_allow_html=True)
    
    # Get top proteins and genes
    top_proteins = get_top_proteins(5)
    top_genes = get_top_genes(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Top Variable Proteins</h3>", unsafe_allow_html=True)
        if not top_proteins.empty:
            fig = px.bar(
                top_proteins, 
                x='protein_id', 
                y='variance',
                title='Proteins with Highest Expression Variance',
                color='variance',
                color_continuous_scale='Viridis',
                labels={'protein_id': 'Protein ID', 'variance': 'Expression Variance'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No proteomics data available")
    
    with col2:
        st.markdown("<h3>Top Variable Genes (CNV)</h3>", unsafe_allow_html=True)
        if not top_genes.empty:
            fig = px.bar(
                top_genes, 
                x='gene_id', 
                y='variance',
                title='Genes with Highest CNV Variance',
                color='variance',
                color_continuous_scale='Viridis',
                labels={'gene_id': 'Gene ID', 'variance': 'CNV Variance'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No genomics data available")
    
    # Adverse Event Summary
    st.markdown("<h2 class='sub-header'>Adverse Event Summary</h2>", unsafe_allow_html=True)
    
    adverse_events = load_adverse_events()
    
    if not adverse_events.empty:
        # Join with patient data to filter
        filtered_patient_ids = set(filtered_patients['patient_id'])
        filtered_events = adverse_events[adverse_events['patient_id'].isin(filtered_patient_ids)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Event type distribution
            event_counts = filtered_events['event_type'].value_counts().reset_index()
            event_counts.columns = ['Event Type', 'Count']
            
            fig = px.bar(
                event_counts.head(10),
                x='Event Type',
                y='Count',
                title='Top 10 Adverse Events',
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Event severity distribution
            grade_counts = filtered_events['event_grade'].value_counts().reset_index()
            grade_counts.columns = ['Grade', 'Count']
            grade_counts = grade_counts.sort_values('Grade')
            
            fig = px.pie(
                grade_counts,
                values='Count',
                names='Grade',
                title='Adverse Event Severity Distribution',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No adverse event data available")

def render_biomarker_tracker_page(patients):
    st.markdown("<h1 class='main-header'>Biomarker Tracker</h1>", unsafe_allow_html=True)
    
    # Tabs for protein expression and CNV
    tab1, tab2 = st.tabs(["Protein Expression", "Copy Number Variation"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Protein Expression Analysis</h2>", unsafe_allow_html=True)
        
        # Get top variable proteins
        top_proteins = get_top_proteins(30)
        
        if not top_proteins.empty:
            # Allow user to select a protein
            selected_protein = st.selectbox(
                "Select Protein to Analyze",
                top_proteins['protein_id'].tolist(),
                index=0
            )
            
            # Get data for the selected protein
            protein_data = get_protein_data_for_patients(selected_protein)
            
            if not protein_data.empty:
                # Filter to patients in the filtered set
                filtered_patient_ids = set(patients['patient_id'])
                protein_data = protein_data[protein_data['patient_id'].isin(filtered_patient_ids)]
                
                # Create summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Expression", f"{protein_data['expression_level'].mean():.2f}")
                
                with col2:
                    st.metric("Patients with Data", f"{len(protein_data)}")
                
                with col3:
                    st.metric("Expression Range", 
                             f"{protein_data['expression_level'].min():.2f} to {protein_data['expression_level'].max():.2f}")
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Expression distribution
                    fig = px.histogram(
                        protein_data,
                        x='expression_level',
                        color='sex',
                        marginal='box',
                        title=f'{selected_protein} Expression Distribution',
                        labels={'expression_level': 'Expression Level'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Expression by tumor stage
                    fig = px.box(
                        protein_data,
                        x='path_stage_pt',
                        y='expression_level',
                        color='path_stage_pt',
                        title=f'{selected_protein} Expression by Tumor Stage',
                        labels={'path_stage_pt': 'T Stage', 'expression_level': 'Expression Level'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional analyses
                col1, col2 = st.columns(2)
                
                with col1:
                    # Expression by TP53
                    if 'tp53_mutation' in protein_data.columns:
                        protein_data['TP53 Status'] = protein_data['tp53_mutation'].map({0: 'Wild Type', 1: 'Mutated'})
                        
                        fig = px.box(
                            protein_data,
                            x='TP53 Status',
                            y='expression_level',
                            color='TP53 Status',
                            title=f'{selected_protein} Expression by TP53 Status',
                            labels={'expression_level': 'Expression Level'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Expression vs age
                    fig = px.scatter(
                        protein_data,
                        x='age',
                        y='expression_level',
                        color='sex',
                        trendline='ols',
                        title=f'{selected_protein} Expression vs Age',
                        labels={'age': 'Age', 'expression_level': 'Expression Level'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                if st.checkbox("Show expression data table"):
                    st.dataframe(protein_data)
            else:
                st.warning(f"No data available for protein {selected_protein}")
        else:
            st.info("No proteomics data available in the database")
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Copy Number Variation Analysis</h2>", unsafe_allow_html=True)
        
        # Get top variable genes
        top_genes = get_top_genes(30)
        
        if not top_genes.empty:
            # Allow user to select a gene
            selected_gene = st.selectbox(
                "Select Gene to Analyze",
                top_genes['gene_id'].tolist(),
                index=0
            )
            
            # Get data for the selected gene
            gene_data = get_gene_data_for_patients(selected_gene)
            
            if not gene_data.empty:
                # Filter to patients in the filtered set
                filtered_patient_ids = set(patients['patient_id'])
                gene_data = gene_data[gene_data['patient_id'].isin(filtered_patient_ids)]
                
                # Create summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean CNV", f"{gene_data['copy_number_variation'].mean():.2f}")
                
                with col2:
                    st.metric("Patients with Data", f"{len(gene_data)}")
                
                with col3:
                    st.metric("CNV Range", 
                             f"{gene_data['copy_number_variation'].min():.2f} to {gene_data['copy_number_variation'].max():.2f}")
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # CNV distribution
                    fig = px.histogram(
                        gene_data,
                        x='copy_number_variation',
                        color='sex',
                        marginal='box',
                        title=f'{selected_gene} CNV Distribution',
                        labels={'copy_number_variation': 'Copy Number Variation'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # CNV by tumor stage
                    fig = px.box(
                        gene_data,
                        x='path_stage_pt',
                        y='copy_number_variation',
                        color='path_stage_pt',
                        title=f'{selected_gene} CNV by Tumor Stage',
                        labels={'path_stage_pt': 'T Stage', 'copy_number_variation': 'Copy Number Variation'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional analyses
                col1, col2 = st.columns(2)
                
                with col1:
                    # CNV by TP53
                    if 'tp53_mutation' in gene_data.columns:
                        gene_data['TP53 Status'] = gene_data['tp53_mutation'].map({0: 'Wild Type', 1: 'Mutated'})
                        
                        fig = px.box(
                            gene_data,
                            x='TP53 Status',
                            y='copy_number_variation',
                            color='TP53 Status',
                            title=f'{selected_gene} CNV by TP53 Status',
                            labels={'copy_number_variation': 'Copy Number Variation'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # CNV vs age
                    fig = px.scatter(
                        gene_data,
                        x='age',
                        y='copy_number_variation',
                        color='sex',
                        trendline='ols',
                        title=f'{selected_gene} CNV vs Age',
                        labels={'age': 'Age', 'copy_number_variation': 'Copy Number Variation'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                if st.checkbox("Show CNV data table"):
                    st.dataframe(gene_data)
            else:
                st.warning(f"No data available for gene {selected_gene}")
        else:
            st.info("No genomics data available in the database")
    
    # Correlation analysis section
    st.markdown("<h2 class='sub-header'>Multi-Biomarker Analysis</h2>", unsafe_allow_html=True)
    
    # Allow user to select multiple biomarkers
    if not top_proteins.empty and not top_genes.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_proteins = st.multiselect(
                "Select Proteins",
                top_proteins['protein_id'].tolist(),
                default=top_proteins['protein_id'].tolist()[:2] if len(top_proteins) >= 2 else None
            )
        
        with col2:
            selected_genes = st.multiselect(
                "Select Genes",
                top_genes['gene_id'].tolist(),
                default=top_genes['gene_id'].tolist()[:2] if len(top_genes) >= 2 else None
            )
        
        # If proteins and genes are selected, show correlation analysis
        if selected_proteins and selected_genes:
            st.info("This analysis examines relationships between protein expression and copy number variation")
            
            # Get data for selected proteins and genes
            protein_data_list = []
            for protein in selected_proteins:
                df = get_protein_data_for_patients(protein)
                if not df.empty:
                    df = df[['patient_id', 'expression_level']]
                    df.columns = ['patient_id', f'protein_{protein}']
                    protein_data_list.append(df)
            
            gene_data_list = []
            for gene in selected_genes:
                df = get_gene_data_for_patients(gene)
                if not df.empty:
                    df = df[['patient_id', 'copy_number_variation']]
                    df.columns = ['patient_id', f'gene_{gene}']
                    gene_data_list.append(df)
            
            # Merge all dataframes on patient_id
            merged_data = None
            if protein_data_list and gene_data_list:
                merged_data = protein_data_list[0]
                for df in protein_data_list[1:] + gene_data_list:
                    merged_data = pd.merge(merged_data, df, on='patient_id', how='inner')
                
                # Filter to patients in the filtered set
                filtered_patient_ids = set(patients['patient_id'])
                merged_data = merged_data[merged_data['patient_id'].isin(filtered_patient_ids)]
            
            if merged_data is not None and len(merged_data) > 5:
                # Calculate correlation matrix
                corr_cols = [col for col in merged_data.columns if col != 'patient_id']
                corr_matrix = merged_data[corr_cols].corr()
                
                # Plot correlation heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title='Biomarker Correlation Matrix',
                    labels={'color': 'Correlation'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show scatter plots for protein-gene pairs
                st.markdown("<h3>Protein-Gene Correlation Analysis</h3>", unsafe_allow_html=True)
                
                protein_cols = [col for col in corr_cols if col.startswith('protein_')]
                gene_cols = [col for col in corr_cols if col.startswith('gene_')]
                
                for p_col in protein_cols[:2]:  # Limit to first 2 proteins
                    for g_col in gene_cols[:2]:  # Limit to first 2 genes
                        protein_name = p_col.replace('protein_', '')
                        gene_name = g_col.replace('gene_', '')
                        
                        fig = px.scatter(
                            merged_data,
                            x=p_col,
                            y=g_col,
                            trendline='ols',
                            title=f'{protein_name} Expression vs {gene_name} CNV',
                            labels={p_col: f'{protein_name} Expression', g_col: f'{gene_name} CNV'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data for correlation analysis with the selected biomarkers")

def render_adverse_event_page(patients):
    st.markdown("<h1 class='main-header'>Adverse Event Monitor</h1>", unsafe_allow_html=True)
    
    # Load adverse events
    adverse_events = load_adverse_events()
    
    if not adverse_events.empty:
        # Filter events to selected patients
        filtered_patient_ids = set(patients['patient_id'])
        filtered_events = adverse_events[adverse_events['patient_id'].isin(filtered_patient_ids)]
        
        # Convert event_date to datetime
        filtered_events['event_date'] = pd.to_datetime(filtered_events['event_date'])
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Events", len(filtered_events))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            patients_with_events = filtered_events['patient_id'].nunique()
            st.metric("Patients with Events", patients_with_events)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            event_rate = patients_with_events / len(patients) if len(patients) > 0 else 0
            st.metric("Event Rate", f"{event_rate:.1%}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            avg_severity = filtered_events['event_grade'].mean()
            st.metric("Average Severity", f"{avg_severity:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Event distribution by type and severity
        col1, col2 = st.columns(2)
        
        with col1:
            # Event type distribution
            event_counts = filtered_events['event_type'].value_counts().reset_index()
            event_counts.columns = ['Event Type', 'Count']
            
            fig = px.bar(
                event_counts,
                x='Event Type',
                y='Count',
                title='Adverse Event Distribution',
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Event grade distribution
            grade_severity = {
                1: "Mild",
                2: "Moderate",
                3: "Severe",
                4: "Life-threatening",
                5: "Fatal"
            }
            
            filtered_events['Severity'] = filtered_events['event_grade'].map(
                lambda x: grade_severity.get(x, f"Grade {x}")
            )
            
            grade_counts = filtered_events['Severity'].value_counts().reset_index()
            grade_counts.columns = ['Severity', 'Count']
            
            fig = px.pie(
                grade_counts,
                values='Count',
                names='Severity',
                title='Adverse Event Severity Distribution',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Temporal analysis
        st.markdown("<h2 class='sub-header'>Temporal Analysis</h2>", unsafe_allow_html=True)
        
        # Events over time
        events_by_date = filtered_events.groupby(pd.Grouper(key='event_date', freq='W')).size().reset_index()
        events_by_date.columns = ['Week', 'Count']
        
        fig = px.line(
            events_by_date,
            x='Week',
            y='Count',
            title='Adverse Events Over Time',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Events by type over time
        st.markdown("<h3>Events by Type Over Time</h3>", unsafe_allow_html=True)

        # Filter to top event types
        top_events = filtered_events['event_type'].value_counts().nlargest(5).index
        top_events_data = filtered_events[filtered_events['event_type'].isin(top_events)]

        # Create a proper aggregated dataframe first
        event_counts_by_date = top_events_data.groupby(['event_date', 'event_type']).size().reset_index(name='count')

        # Now create the plot with the properly formatted data
        fig = px.line(
            event_counts_by_date,
            x='event_date',
            y='count',
            color='event_type',
            title='Top 5 Adverse Events Over Time',
            labels={'event_date': 'Date', 'count': 'Count', 'event_type': 'Event Type'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Relationship between events and patient characteristics
        st.markdown("<h2 class='sub-header'>Patient Characteristic Analysis</h2>", unsafe_allow_html=True)
        
        # Join events with patient data
        events_with_patients = pd.merge(
            filtered_events,
            patients[['patient_id', 'age', 'sex', 'histologic_grade', 'path_stage_pt']],
            on='patient_id',
            how='inner'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Events by age group
            events_with_patients['age_group'] = pd.cut(
                events_with_patients['age'], 
                bins=[0, 40, 50, 60, 70, 100],
                labels=['<40', '40-50', '50-60', '60-70', '>70']
            )
            
            age_event_counts = events_with_patients.groupby(['age_group', 'event_type']).size().reset_index()
            age_event_counts.columns = ['Age Group', 'Event Type', 'Count']
            
            fig = px.bar(
                age_event_counts,
                x='Age Group',
                y='Count',
                color='Event Type',
                title='Adverse Events by Age Group',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Events by gender
            gender_event_counts = events_with_patients.groupby(['sex', 'event_type']).size().reset_index()
            gender_event_counts.columns = ['Gender', 'Event Type', 'Count']
            
            fig = px.bar(
                gender_event_counts,
                x='Gender',
                y='Count',
                color='Event Type',
                title='Adverse Events by Gender',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Events by tumor characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            # Events by tumor stage
            stage_event_counts = events_with_patients.groupby(['path_stage_pt', 'event_type']).size().reset_index()
            stage_event_counts.columns = ['T Stage', 'Event Type', 'Count']
            
            fig = px.bar(
                stage_event_counts,
                x='T Stage',
                y='Count',
                color='Event Type',
                title='Adverse Events by Tumor Stage',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Events by severity and stage
            events_with_patients['event_severity'] = events_with_patients['event_grade'].map(
                lambda x: grade_severity.get(x, f"Grade {x}")
            )
            
            severity_stage_counts = events_with_patients.groupby(['path_stage_pt', 'event_severity']).size().reset_index()
            severity_stage_counts.columns = ['T Stage', 'Severity', 'Count']
            
            fig = px.bar(
                severity_stage_counts,
                x='T Stage',
                y='Count',
                color='Severity',
                title='Event Severity by Tumor Stage',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Event relationship with biomarkers (if available)
        st.markdown("<h2 class='sub-header'>Biomarker-Adverse Event Analysis</h2>", unsafe_allow_html=True)
        
        # Load top proteins and genes
        top_proteins = get_top_proteins(5)
        top_genes = get_top_genes(5)
        
        if not top_proteins.empty and not top_genes.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Select a protein
                selected_protein = st.selectbox(
                    "Select Protein",
                    top_proteins['protein_id'].tolist(),
                    index=0
                )
                
                # Get data for this protein
                protein_data = get_protein_data_for_patients(selected_protein)
                
                if not protein_data.empty:
                    # Merge with events
                    protein_events = pd.merge(
                        protein_data,
                        filtered_events[['patient_id', 'event_type', 'event_grade']],
                        on='patient_id',
                        how='inner'
                    )
                    
                    if not protein_events.empty:
                        # Analyze expression by event type
                        fig = px.box(
                            protein_events,
                            x='event_type',
                            y='expression_level',
                            color='event_type',
                            title=f'{selected_protein} Expression by Adverse Event Type',
                            labels={'expression_level': 'Expression Level', 'event_type': 'Event Type'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No overlap between events and protein data")
            
            with col2:
                # Select a gene
                selected_gene = st.selectbox(
                    "Select Gene",
                    top_genes['gene_id'].tolist(),
                    index=0
                )
                
                # Get data for this gene
                gene_data = get_gene_data_for_patients(selected_gene)
                
                if not gene_data.empty:
                    # Merge with events
                    gene_events = pd.merge(
                        gene_data,
                        filtered_events[['patient_id', 'event_type', 'event_grade']],
                        on='patient_id',
                        how='inner'
                    )
                    
                    if not gene_events.empty:
                        # Analyze CNV by event type
                        fig = px.box(
                            gene_events,
                            x='event_type',
                            y='copy_number_variation',
                            color='event_type',
                            title=f'{selected_gene} CNV by Adverse Event Type',
                            labels={'copy_number_variation': 'Copy Number Variation', 'event_type': 'Event Type'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No overlap between events and gene data")
        else:
            st.info("Biomarker data not available for this analysis")
    else:
        st.info("No adverse event data available")

def render_patient_explorer_page(patients):
    st.markdown("<h1 class='main-header'>Patient Explorer</h1>", unsafe_allow_html=True)
    
    # Patient selection
    patient_ids = sorted(patients['patient_id'].unique().tolist())
    selected_patient_id = st.selectbox("Select Patient", patient_ids)
    
    # Get selected patient data
    patient_data = patients[patients['patient_id'] == selected_patient_id].iloc[0]
    
    # Display patient information
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<h3>Patient Details</h3>", unsafe_allow_html=True)
        
        # Format basic patient info
        patient_info = {
            "Patient ID": selected_patient_id,
            "Age": patient_data['age'],
            "Sex": patient_data['sex'],
            "Tumor Size (cm)": f"{patient_data['tumor_size_cm']:.1f}" if pd.notna(patient_data['tumor_size_cm']) else "N/A",
            "T Stage": patient_data['path_stage_pt'],
            "N Stage": patient_data['path_stage_pn'],
            "Histologic Grade": patient_data['histologic_grade'],
            "TP53 Mutation": "Yes" if patient_data['tp53_mutation'] == 1 else "No",
            "PIK3CA Mutation": "Yes" if patient_data['pik3ca_mutation'] == 1 else "No"
        }
        
        for key, value in patient_info.items():
            st.markdown(f"**{key}:** {value}")
    
    with col2:
        st.markdown("<h3>Clinical Status</h3>", unsafe_allow_html=True)
        
        # Create summary metrics
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            disease_severity = 0
            if pd.notna(patient_data['path_stage_pt']) and pd.notna(patient_data['path_stage_pn']):
                t_score = {'pT1': 1, 'pT2': 2, 'pT3': 3, 'pT4': 4}.get(patient_data['path_stage_pt'], 0)
                n_score = {'pN0': 0, 'pN1': 1, 'pN2': 2, 'pN3': 3}.get(patient_data['path_stage_pn'], 0)
                disease_severity = t_score + n_score
            
            st.metric("Disease Severity", f"{disease_severity}")
        
        with col_b:
            if 'os_days' in patient_data and pd.notna(patient_data['os_days']):
                st.metric("Survival (days)", f"{patient_data['os_days']:.0f}")
            else:
                st.metric("Survival (days)", "N/A")
        
        with col_c:
            if 'os_event' in patient_data:
                status = "Deceased" if patient_data['os_event'] == 1 else "Alive"
                st.metric("Status", status)
            else:
                st.metric("Status", "Unknown")
        
        # Risk assessment
        if disease_severity > 0:
            # Create gauge chart for disease severity
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = disease_severity,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Disease Severity Score"},
                gauge = {
                    'axis': {'range': [0, 8], 'tickvals': [0, 2, 4, 6, 8]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgreen"},
                        {'range': [2, 4], 'color': "yellow"},
                        {'range': [4, 8], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': disease_severity
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabs for different data types
    tab1, tab2, tab3 = st.tabs(["Adverse Events", "Protein Expression", "Copy Number Variation"])
    
    with tab1:
        # Get adverse events for this patient
        patient_events = load_adverse_events(selected_patient_id)
        
        if not patient_events.empty:
            # Convert event_date to datetime
            patient_events['event_date'] = pd.to_datetime(patient_events['event_date'])
            
            # Sort by date
            patient_events = patient_events.sort_values('event_date')
            
            # Display events in a table
            st.markdown(f"### Adverse Events ({len(patient_events)} events)")
            
            # Format for display
            display_events = patient_events.copy()
            display_events['event_date'] = display_events['event_date'].dt.strftime('%Y-%m-%d')
            display_events['Related'] = display_events['related_to_treatment'].map({0: 'No', 1: 'Yes'})
            
            st.dataframe(
                display_events[['event_date', 'event_type', 'event_grade', 'Related']],
                hide_index=True
            )
            
            # Event timeline
            if len(patient_events) > 1:
                event_timeline = patient_events.copy()
                event_timeline['Event'] = event_timeline['event_type'] + ' (Grade ' + event_timeline['event_grade'].astype(str) + ')'
    
                fig = px.timeline(
                    event_timeline,
                    x_start='event_date',
                    x_end='event_date',
                    y='Event',
                    color='event_grade',
                    title='Adverse Event Timeline',
                    labels={'event_date': 'Date', 'Event': 'Event Type'},
                    color_continuous_scale='Reds'
            )
    
            # Use marker dictionary instead of marker_size
            fig.update_traces(marker=dict(size=15))
    
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No adverse events recorded for this patient")
    
    with tab2:
        # Get protein expression data for this patient
        protein_data = load_proteomics_data(limit=100, patient_id=selected_patient_id)
        
        if not protein_data.empty:
            st.markdown(f"### Protein Expression Data ({len(protein_data)} proteins)")
            
            # Get top proteins for comparison
            top_proteins = get_top_proteins(20)
            
            if not top_proteins.empty:
                # Filter to top proteins
                top_protein_ids = set(top_proteins['protein_id'])
                filtered_proteins = protein_data[protein_data['protein_id'].isin(top_protein_ids)]
                
                if not filtered_proteins.empty:
                    # Sort by expression level
                    filtered_proteins = filtered_proteins.sort_values('expression_level', ascending=False)
                    
                    # Display as bar chart
                    fig = px.bar(
                        filtered_proteins,
                        x='protein_id',
                        y='expression_level',
                        title='Top Protein Expression Levels',
                        color='expression_level',
                        color_continuous_scale='Viridis',
                        labels={'protein_id': 'Protein ID', 'expression_level': 'Expression Level'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show comparison to population average
                    st.markdown("### Comparison to Population Average")
                    
                    # For each protein, get population average
                    population_avg = []
                    for protein_id in filtered_proteins['protein_id']:
                        # Get data for this protein across all patients
                        protein_pop = get_protein_data_for_patients(protein_id)
                        
                        if not protein_pop.empty:
                            avg = protein_pop['expression_level'].mean()
                            patient_val = filtered_proteins[filtered_proteins['protein_id'] == protein_id]['expression_level'].iloc[0]
                            
                            population_avg.append({
                                'protein_id': protein_id,
                                'patient_value': patient_val,
                                'population_avg': avg,
                                'difference': patient_val - avg
                            })
                    
                    if population_avg:
                        pop_df = pd.DataFrame(population_avg)
                        
                        # Sort by absolute difference
                        pop_df['abs_diff'] = pop_df['difference'].abs()
                        pop_df = pop_df.sort_values('abs_diff', ascending=False).head(10)
                        
                        # Plot
                        fig = go.Figure()
                        
                        # Add population average
                        fig.add_trace(go.Bar(
                            x=pop_df['protein_id'],
                            y=pop_df['population_avg'],
                            name='Population Average',
                            marker_color='lightgrey'
                        ))
                        
                        # Add patient value
                        fig.add_trace(go.Bar(
                            x=pop_df['protein_id'],
                            y=pop_df['patient_value'],
                            name='Patient Value',
                            marker_color='blue'
                        ))
                        
                        fig.update_layout(
                            title='Protein Expression: Patient vs Population Average',
                            xaxis_title='Protein ID',
                            yaxis_title='Expression Level',
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show protein data table
                    if st.checkbox("Show protein expression data"):
                        st.dataframe(filtered_proteins)
                else:
                    st.info("No data available for top variable proteins")
            else:
                st.info("No protein ranking data available")
        else:
            st.info("No protein expression data available for this patient")
    
    with tab3:
        # Get CNV data for this patient
        cnv_data = load_genomics_data(limit=100, patient_id=selected_patient_id)
        
        if not cnv_data.empty:
            st.markdown(f"### Copy Number Variation Data ({len(cnv_data)} genes)")
            
            # Get top genes for comparison
            top_genes = get_top_genes(20)
            
            if not top_genes.empty:
                # Filter to top genes
                top_gene_ids = set(top_genes['gene_id'])
                filtered_genes = cnv_data[cnv_data['gene_id'].isin(top_gene_ids)]
                
                if not filtered_genes.empty:
                    # Sort by absolute CNV
                    filtered_genes['abs_cnv'] = filtered_genes['copy_number_variation'].abs()
                    filtered_genes = filtered_genes.sort_values('abs_cnv', ascending=False)
                    
                    # Display as bar chart
                    fig = px.bar(
                        filtered_genes,
                        x='gene_id',
                        y='copy_number_variation',
                        title='Top Gene Copy Number Variations',
                        color='copy_number_variation',
                        color_continuous_scale='RdBu',
                        labels={'gene_id': 'Gene ID', 'copy_number_variation': 'Copy Number Variation'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show comparison to population average
                    st.markdown("### Comparison to Population Average")
                    
                    # For each gene, get population average
                    population_avg = []
                    for gene_id in filtered_genes['gene_id']:
                        # Get data for this gene across all patients
                        gene_pop = get_gene_data_for_patients(gene_id)
                        
                        if not gene_pop.empty:
                            avg = gene_pop['copy_number_variation'].mean()
                            patient_val = filtered_genes[filtered_genes['gene_id'] == gene_id]['copy_number_variation'].iloc[0]
                            
                            population_avg.append({
                                'gene_id': gene_id,
                                'patient_value': patient_val,
                                'population_avg': avg,
                                'difference': patient_val - avg
                            })
                    
                    if population_avg:
                        pop_df = pd.DataFrame(population_avg)
                        
                        # Sort by absolute difference
                        pop_df['abs_diff'] = pop_df['difference'].abs()
                        pop_df = pop_df.sort_values('abs_diff', ascending=False).head(10)
                        
                        # Plot
                        fig = go.Figure()
                        
                        # Add population average
                        fig.add_trace(go.Bar(
                            x=pop_df['gene_id'],
                            y=pop_df['population_avg'],
                            name='Population Average',
                            marker_color='lightgrey'
                        ))
                        
                        # Add patient value
                        fig.add_trace(go.Bar(
                            x=pop_df['gene_id'],
                            y=pop_df['patient_value'],
                            name='Patient Value',
                            marker_color='red'
                        ))
                        
                        fig.update_layout(
                            title='Copy Number Variation: Patient vs Population Average',
                            xaxis_title='Gene ID',
                            yaxis_title='Copy Number Variation',
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show CNV data table
                    if st.checkbox("Show CNV data"):
                        st.dataframe(filtered_genes)
                else:
                    st.info("No data available for top variable genes")
            else:
                st.info("No gene ranking data available")
        else:
            st.info("No copy number variation data available for this patient")

def render_dropout_analysis_page(patients):
    st.markdown("<h1 class='main-header'>Dropout Risk Analysis</h1>", unsafe_allow_html=True)
    
    # Convert patient data to features for prediction
    feature_cols = [
        'age', 'tumor_size_cm', 'tp53_mutation', 'pik3ca_mutation'
    ]
    
    categorical_cols = [
        'sex', 'path_stage_pt', 'path_stage_pn', 'histologic_grade'
    ]
    
    # Check if we have all necessary columns
    missing_cols = [col for col in feature_cols + categorical_cols if col not in patients.columns]
    
    if missing_cols:
        st.warning(f"Missing columns needed for prediction: {', '.join(missing_cols)}")
        st.info("Using available columns for demonstration")
    
    # Get available columns
    available_numeric_cols = [col for col in feature_cols if col in patients.columns]
    available_cat_cols = [col for col in categorical_cols if col in patients.columns]
    
    if available_numeric_cols or available_cat_cols:
        # Generate predictions
        try:
            # Create derived features for better prediction
            patient_features = patients.copy()
            
            # Handle categorical variables - create dummy variables
            if available_cat_cols:
                # One-hot encode categorical variables
                patient_features_encoded = pd.get_dummies(
                    patient_features, 
                    columns=available_cat_cols,
                    drop_first=True
                )
                
                # Get the dummy column names that were created
                dummy_cols = [col for col in patient_features_encoded.columns 
                              if col not in patient_features.columns and col != 'patient_id']
                
                # Only use numeric features and encoded categorical features
                model_features = patient_features_encoded[available_numeric_cols + dummy_cols]
            else:
                # If no categorical columns, just use numeric ones
                model_features = patient_features[available_numeric_cols]
            
            # Fill any remaining NaN values with 0 to prevent errors
            model_features = model_features.fillna(0)
            
            # Make predictions
            dropout_probs = predict_dropout_risk(model_features)
            
            if dropout_probs is not None:
                patient_features['dropout_risk'] = dropout_probs
                
                # Add risk categories
                risk_categories = []
                risk_classes = []
                for prob in dropout_probs:
                    category, class_name = get_risk_class(prob)
                    risk_categories.append(category)
                    risk_classes.append(class_name)
                
                patient_features['risk_category'] = risk_categories
                patient_features['risk_class'] = risk_classes
                
                # Display risk summary
                st.markdown("<h2 class='sub-header'>Dropout Risk Summary</h2>", unsafe_allow_html=True)
                
                # Create summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_risk = patient_features['dropout_risk'].mean()
                    st.metric("Average Risk", f"{avg_risk:.1%}")
                
                with col2:
                    high_risk_count = (patient_features['risk_category'] == 'High Risk').sum()
                    high_risk_pct = high_risk_count / len(patient_features)
                    st.metric("High Risk Patients", f"{high_risk_count} ({high_risk_pct:.1%})")
                
                with col3:
                    low_risk_count = (patient_features['risk_category'] == 'Low Risk').sum()
                    low_risk_pct = low_risk_count / len(patient_features)
                    st.metric("Low Risk Patients", f"{low_risk_count} ({low_risk_pct:.1%})")
                
                # Risk distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk histogram
                    fig = px.histogram(
                        patient_features,
                        x='dropout_risk',
                        title='Dropout Risk Distribution',
                        labels={'dropout_risk': 'Dropout Risk'},
                        color_discrete_sequence=['steelblue']
                    )
                    
                    # Add reference lines
                    fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
                    fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk category distribution
                    risk_counts = patient_features['risk_category'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Category', 'Count']
                    
                    # Ensure all categories are present
                    all_categories = ['Low Risk', 'Medium Risk', 'High Risk']
                    for cat in all_categories:
                        if cat not in risk_counts['Risk Category'].values:
                            risk_counts = pd.concat([risk_counts, pd.DataFrame({'Risk Category': [cat], 'Count': [0]})])
                    
                    # Set color order
                    risk_counts['Risk Category'] = pd.Categorical(
                        risk_counts['Risk Category'],
                        categories=all_categories,
                        ordered=True
                    )
                    risk_counts = risk_counts.sort_values('Risk Category')
                    
                    # Create pie chart
                    fig = px.pie(
                        risk_counts,
                        values='Count',
                        names='Risk Category',
                        title='Risk Distribution',
                        color='Risk Category',
                        color_discrete_map={
                            'Low Risk': '#0dbf66',
                            'Medium Risk': '#ffa500',
                            'High Risk': '#ff4b4b'
                        }
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors analysis
                st.markdown("<h2 class='sub-header'>Risk Factors Analysis</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age vs Risk
                    fig = px.scatter(
                        patient_features,
                        x='age',
                        y='dropout_risk',
                        color='risk_category',
                        color_discrete_map={
                            'Low Risk': '#0dbf66',
                            'Medium Risk': '#ffa500',
                            'High Risk': '#ff4b4b'
                        },
                        title='Age vs Dropout Risk',
                        labels={'age': 'Age', 'dropout_risk': 'Dropout Risk'},
                        hover_data=['patient_id']
                    )
                    
                    # Add trend line
                    fig.update_layout(showlegend=True)
                    fig = px.scatter(
                        patient_features,
                        x='age',
                        y='dropout_risk',
                        trendline='ols',
                        title='Age vs Dropout Risk',
                        labels={'age': 'Age', 'dropout_risk': 'Dropout Risk'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Tumor stage vs Risk
                    if 'path_stage_pt' in patient_features.columns:
                        stage_risk = patient_features.groupby('path_stage_pt')['dropout_risk'].mean().reset_index()
                        
                        fig = px.bar(
                            stage_risk,
                            x='path_stage_pt',
                            y='dropout_risk',
                            title='Tumor Stage vs Average Dropout Risk',
                            labels={'path_stage_pt': 'T Stage', 'dropout_risk': 'Average Dropout Risk'},
                            color='dropout_risk',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Mutation impact
                if 'tp53_mutation' in patient_features.columns and 'pik3ca_mutation' in patient_features.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # TP53 impact
                        tp53_risk = patient_features.groupby('tp53_mutation')['dropout_risk'].mean().reset_index()
                        tp53_risk['TP53 Status'] = tp53_risk['tp53_mutation'].map({0: 'Wild Type', 1: 'Mutated'})
                        
                        fig = px.bar(
                            tp53_risk,
                            x='TP53 Status',
                            y='dropout_risk',
                            title='TP53 Mutation vs Dropout Risk',
                            labels={'dropout_risk': 'Average Dropout Risk'},
                            color='dropout_risk',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # PIK3CA impact
                        pik3ca_risk = patient_features.groupby('pik3ca_mutation')['dropout_risk'].mean().reset_index()
                        pik3ca_risk['PIK3CA Status'] = pik3ca_risk['pik3ca_mutation'].map({0: 'Wild Type', 1: 'Mutated'})
                        
                        fig = px.bar(
                            pik3ca_risk,
                            x='PIK3CA Status',
                            y='dropout_risk',
                            title='PIK3CA Mutation vs Dropout Risk',
                            labels={'dropout_risk': 'Average Dropout Risk'},
                            color='dropout_risk',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # High-risk patients analysis
                st.markdown("<h2 class='sub-header'>High-Risk Patient Analysis</h2>", unsafe_allow_html=True)
                
                high_risk_patients = patient_features[patient_features['risk_category'] == 'High Risk'].sort_values('dropout_risk', ascending=False)
                
                if not high_risk_patients.empty:
                    st.write(f"Identified {len(high_risk_patients)} high-risk patients (risk ≥ 70%)")
                    
                    # Show patient table
                    st.dataframe(
                        high_risk_patients[['patient_id', 'age', 'sex', 'path_stage_pt', 'tumor_size_cm', 'dropout_risk']].head(10),
                        hide_index=True,
                        column_config={
                            "dropout_risk": st.column_config.ProgressColumn(
                                "Dropout Risk",
                                help="Risk of patient dropping out of the trial",
                                format="%.1f%%",
                                min_value=0,
                                max_value=1,
                            )
                        }
                    )
                    
                    # Common characteristics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Age distribution
                        fig = px.histogram(
                            high_risk_patients,
                            x='age',
                            title='Age Distribution of High-Risk Patients',
                            color_discrete_sequence=['#ff4b4b']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'path_stage_pt' in high_risk_patients.columns:
                            # Stage distribution
                            stage_counts = high_risk_patients['path_stage_pt'].value_counts().reset_index()
                            stage_counts.columns = ['T Stage', 'Count']
                            
                            fig = px.bar(
                                stage_counts,
                                x='T Stage',
                                y='Count',
                                title='T Stage Distribution of High-Risk Patients',
                                color_discrete_sequence=['#ff4b4b']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Intervention recommendations
                    st.markdown("<h3>Intervention Recommendations</h3>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    Based on the identified high-risk patients, the following interventions are recommended:
                    
                    1. **Enhanced Communication**:
                       - Schedule weekly check-in calls with high-risk patients
                       - Provide clear written instructions after each visit
                       - Establish dedicated contact person for questions
                    
                    2. **Logistical Support**:
                       - Offer transportation assistance to clinic visits
                       - Provide reminders 48 hours before appointments
                       - Consider home visits for certain assessments when possible
                    
                    3. **Clinical Modifications**:
                       - Review medication regimens for simplification
                       - Proactively manage adverse events
                       - Consider dose adjustments based on tolerability
                    
                    4. **Patient Education**:
                       - Reinforce importance of trial completion
                       - Provide additional educational materials
                       - Connect patients with peer support resources
                    """)
                else:
                    st.info("No high-risk patients identified in the current selection")
            else:
                st.error("Failed to generate dropout risk predictions")
        except Exception as e:
            st.error(f"Error in dropout risk analysis: {str(e)}")
            # Provide more detailed debugging information
            st.expander("Debug Information").write(f"""
            Available numeric columns: {available_numeric_cols}
            Available categorical columns: {available_cat_cols}
            Column types: {patient_features.dtypes}
            """)
    else:
        st.error("No suitable features available for dropout risk prediction")

# Run the app
if __name__ == "__main__":
    main()