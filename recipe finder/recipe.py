
import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("recipe_dataset.csv")

# Train TF-IDF Vectorizer for NLP-based search
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["Ingredients"].fillna(""))

def find_recipes(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X).flatten()
    results = df.iloc[similarity.argsort()[::-1][:10]]
    return results

# Sidebar with custom styling
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #ff7f50;
            color: white;
            font-size: 18px;
        }
        .css-1aumxhk, .css-1v3fvcr {
            font-size: 20px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🔍 Recipe Navigator")
page = st.sidebar.radio("Go to", ["Home", "Recipe Finder", "About", "Settings"], key="nav")

if page == "Home":
    st.title("🍽️ Welcome to Recipe Finder!")
    st.write("Discover delicious recipes based on ingredients and preferences.")
    st.image("download.jpg", use_column_width=True)
    
    st.subheader("🍴 Categories")
    for category in df["Category"].unique():
        st.markdown(f"- {category}")

elif page == "Recipe Finder":
    st.title("🔍 Find Your Perfect Recipe")
    
    # NLP-powered search
    search_term = st.text_input("Enter an ingredient or recipe name:")
    
    if search_term:
        results = find_recipes(search_term)
        
        if not results.empty:
            st.write(f"### Found {len(results)} best-matching recipes:")
            st.dataframe(results)
        else:
            st.write("No matching recipes found.")
    
elif page == "About":
    st.title("ℹ️ About Recipe Finder")
    st.write("This app helps users find recipes using NLP-based search and category filters.")
    st.write("Built with ❤️ using Streamlit and NLP models.")

elif page == "Settings":
    st.title("⚙️ Settings")
    theme = st.radio("Choose a theme:", ["Light", "Dark"], key="theme")
    
    if theme == "Dark":
        st.markdown("""<style>body { background-color: #333; color: white; }</style>""", unsafe_allow_html=True)
        st.write("Dark mode activated!")
    else:
        st.markdown("""<style>body { background-color: white; color: black; }</style>""", unsafe_allow_html=True)
        st.write("Light mode activated!")
