import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the preprocessed DataFrame and TF-IDF vectorizer
with open("processed_dataframe.pkl", "rb") as df_file:
    df = pickle.load(df_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to recommend tests and diseases
def recommend_tests_and_disease(user_symptoms, data, vectorizer):
    # Vectorize the symptoms using the loaded TF-IDF vectorizer
    tfidf_matrix = vectorizer.fit_transform(data["Symptoms"])
    user_tfidf = vectorizer.transform([user_symptoms])
    
    # Compute similarity between user input and dataset entries
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Find the most similar entry
    best_match_idx = similarities.argmax()
    
    # Fetch recommendations from the dataset
    recommended_tests = data.loc[best_match_idx, "Tests"]
    probable_disease = data.loc[best_match_idx, "Disease"]
    
    return recommended_tests, probable_disease

# Streamlit front-end interface
st.title("Medical Recommendation System")

st.header("Describe your symptoms")
user_input = st.text_area("Enter your symptoms here:", 
                         "E.g., I have a fever, cough, and body aches.")

if st.button("Get Recommendation"):
    if user_input.strip():
        try:
            # Get recommendations
            recommended_tests, probable_disease = recommend_tests_and_disease(user_input, df, vectorizer)
            
            # Display results
            st.subheader("Recommended Tests:")
            st.write(recommended_tests)
            
            st.subheader("Probable Disease:")
            st.write(probable_disease)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter some symptoms.")
