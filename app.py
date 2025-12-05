import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from classify_comments import CommentCategorizer
from response_templates import get_response_template, RESPONSE_TEMPLATES
import json
import io

st.set_page_config(
    page_title="Comment Categorization Tool",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

if 'categorizer' not in st.session_state:
    try:
        st.session_state.categorizer = CommentCategorizer()
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.error = str(e)

st.markdown('<h1 class="main-header">💬 Comment Categorization & Reply Assistant</h1>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio(
        "Choose a page:",
        ["🏠 Home", "📊 Batch Analysis"]
    )
    
    st.markdown("---")
    st.header("ℹ️ Model Info")
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully")
        st.caption("Ready to classify comments")
    else:
        st.error("❌ Model not loaded")
        st.caption(f"Error: {st.session_state.get('error', 'Unknown error')}")
        st.caption("Please train the model first using train_model.py")

if page == "🏠 Home":
    st.header("Single Comment Classification")
    st.markdown("Enter a comment below to classify it and get a suggested response.")
    
    if 'comment_text' not in st.session_state:
        st.session_state.comment_text = ""
    
    comment_text = st.text_area(
        "Enter your comment:",
        value=st.session_state.comment_text,
        height=100,
        placeholder="e.g., Amazing work! The animation looks stunning.",
        key="comment_input"
    )
    
    st.session_state.comment_text = comment_text
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        classify_btn = st.button("🔍 Classify Comment", type="primary", width='stretch')
    
    with col2:
        clear_btn = st.button("🗑️ Clear", width='stretch')
    
    if clear_btn:
        st.session_state.comment_text = ""
        st.rerun()
    
    if classify_btn and comment_text:
        if not st.session_state.model_loaded:
            st.error("Model not loaded. Please train the model first.")
        else:
            with st.spinner("Classifying comment..."):
                try:
                    category, confidence = st.session_state.categorizer.classify(comment_text)
                    
                    st.markdown("---")
                    st.subheader("📊 Classification Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Category", category)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    st.progress(confidence)
                    
                    st.markdown("---")
                    st.subheader("💬 Suggested Response")
                    
                    response = get_response_template(category)
                    st.info(response)
                    
                    st.code(response, language=None)
                    
                except Exception as e:
                    st.error(f"Error classifying comment: {str(e)}")

elif page == "📊 Batch Analysis":
    st.header("Batch Comment Analysis")
    st.markdown("Upload a CSV or JSON file with comments to classify them in bulk.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'json'],
        help="Upload a CSV file with a 'comment' column or a JSON file with comment objects"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} comments from CSV")
                
                st.subheader("📄 File Preview")
                st.dataframe(df.head(), width='stretch')
                
                if 'comment' not in df.columns:
                    st.warning("⚠️ No 'comment' column found. Please select the column containing comments:")
                    comment_col = st.selectbox("Select comment column:", df.columns)
                else:
                    comment_col = 'comment'
                
                if st.button("🚀 Classify All Comments", type="primary"):
                    if not st.session_state.model_loaded:
                        st.error("Model not loaded. Please train the model first.")
                    else:
                        with st.spinner("Classifying comments..."):
                            results = st.session_state.categorizer.classify_batch(df[comment_col].tolist())
                            
                            df['predicted_category'] = [r[0] for r in results]
                            df['confidence'] = [r[1] for r in results]
                            
                            st.session_state.results_df = df
                            st.success("✅ Classification complete!")
            
            elif uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                st.success(f"✅ Loaded JSON file")
                
                if isinstance(data, list):
                    comments = [item.get('comment', '') for item in data]
                else:
                    comments = [data.get('comment', '')]
                
                st.subheader("📄 File Preview")
                st.json(data[:3] if isinstance(data, list) else data)
                
                if st.button("🚀 Classify All Comments", type="primary"):
                    if not st.session_state.model_loaded:
                        st.error("Model not loaded. Please train the model first.")
                    else:
                        with st.spinner("Classifying comments..."):
                            results = st.session_state.categorizer.classify_batch(comments)
                            
                            if isinstance(data, list):
                                for i, (category, confidence) in enumerate(results):
                                    data[i]['predicted_category'] = category
                                    data[i]['confidence'] = confidence
                            else:
                                data['predicted_category'] = results[0][0]
                                data['confidence'] = results[0][1]
                            
                            st.session_state.results_json = data
                            st.success("✅ Classification complete!")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    if 'results_df' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Classification Results")
        
        df = st.session_state.results_df
        
        st.dataframe(df, width='stretch')
        
        st.subheader("📈 Category Distribution")
        category_counts = df['predicted_category'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                labels={'x': 'Category', 'y': 'Count'},
                title="Comments by Category"
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, width='stretch')
        
        with col2:
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Category Distribution"
            )
            st.plotly_chart(fig_pie, width='stretch')
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="classified_comments.csv",
            mime="text/csv"
        )
    
    elif 'results_json' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Classification Results")
        st.json(st.session_state.results_json)
        
        json_str = json.dumps(st.session_state.results_json, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Results as JSON",
            data=json_str,
            file_name="classified_comments.json",
            mime="application/json"
        )

