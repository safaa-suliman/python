import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import shutil  # For clearing temporary files
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF  # Import NMF
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Set page configuration
st.set_page_config(page_title="Document Analysis Webpage", page_icon="📄", layout="wide")
st.subheader("Hi, This is a web for analyzing documents :wave:")
st.title("A Data Analyst From Sudan")
st.write("I am passionate about Data Science")
st.write("[My GitHub >](https://github.com/safa-suliman)")

# Define the function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {e}")
        return ""

# Topic Modeling with NMF
def nmf_topic_modeling(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_nmf = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words_nmf)}")
    return topics

# Clear temporary folder
def clear_temp_folder(folder="temp"):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# Function to perform text analysis
def analyze_texts(pdf_texts, top_n):
    all_text = " ".join([doc["text"] for doc in pdf_texts])

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(re.sub(r'\W+', ' ', all_text.lower()))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)

    return top_words, word_counts

# Topic Modeling with LDA
def topic_modeling(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics

# Clustering using KMeans
def clustering(pdf_texts, num_clusters=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([doc["text"] for doc in pdf_texts])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(tfidf_matrix)
    return kmeans.labels_

# Function to find related words
def find_related_words(texts, user_word, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    if user_word not in feature_names:
        return None, None

    user_word_idx = feature_names.tolist().index(user_word)
    user_word_vector = tfidf_matrix[:, user_word_idx]

    word_similarities = cosine_similarity(user_word_vector.T, tfidf_matrix.T).flatten()
    sorted_indices = word_similarities.argsort()[::-1][1:top_n + 1]
    related_words = [(feature_names[i], word_similarities[i]) for i in sorted_indices]

    return related_words

# Streamlit App
st.title("📂 Document Analysis - Enhanced Features")

uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    clear_temp_folder()
    pdf_texts = []

    for uploaded_file in uploaded_files:
        pdf_path = os.path.join("temp", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                pdf_texts.append({"filename": uploaded_file.name, "text": text})
            else:
                st.warning(f"No text extracted from {uploaded_file.name}.")
        else:
            st.warning(f"Skipping non-PDF file: {uploaded_file.name}")

    if not pdf_texts:
        st.error("No text could be extracted from the uploaded PDFs.")
    else:
        pdf_df = pd.DataFrame(pdf_texts)
        st.write("### Extracted Data:")
        st.dataframe(pdf_df)

        csv_data = pdf_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv_data, file_name="extracted_texts.csv", mime="text/csv")

        top_n = st.slider("Select the number of top words to display", min_value=1, max_value=20, value=10)

        if st.button("Analyze Texts"):
            top_words, word_counts = analyze_texts(pdf_texts, top_n)
            st.session_state.top_words = top_words
            st.session_state.word_counts = word_counts

        if "top_words" in st.session_state:
            st.write("### Top Words Across All Documents:")
            st.table(pd.DataFrame(st.session_state.top_words, columns=["Word", "Frequency"]))

        specific_word = st.text_input("Enter a word to analyze its frequency:")
        if st.button("Calculate Frequency"):
            if specific_word:
                specific_word_count = st.session_state.word_counts.get(specific_word.lower(), 0)
                st.write(f"The word **'{specific_word}'** appears **{specific_word_count}** times.")

        user_word = st.text_input("Enter a word to find related terms:")
        if st.button("Find Related Words"):
            texts_only = [doc["text"] for doc in pdf_texts]
            related_words = find_related_words(texts_only, user_word.lower(), top_n=5)
            if related_words:
                st.write(f"### Words Related to **'{user_word}'**:")
                st.table(pd.DataFrame(related_words, columns=["Word", "Similarity Score"]))

        num_topics = st.slider("Select the Number of Topics:", 2, 10, 3)
        topics_lda = topic_modeling([doc["text"] for doc in pdf_texts], num_topics=num_topics)
        st.write("### LDA Topic Modeling Results:")
        for topic in topics_lda:
            st.write(topic)

        clusters = clustering(pdf_texts, num_clusters=num_topics)
        pdf_df["Cluster"] = clusters
        st.write("### Clustered Documents:")
        st.dataframe(pdf_df)

        plt.figure(figsize=(10, 6))
        sns.countplot(x=pdf_df["Cluster"])
        plt.title("Document Cluster Distribution")
        st.pyplot(plt)
