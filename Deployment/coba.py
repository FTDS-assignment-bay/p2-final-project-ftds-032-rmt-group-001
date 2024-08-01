# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException

# Create the main program
def run():

    # Download necessary NLTK resources
    nltk.download('stopwords')
    nltk.download('punkt')

    # Inisialisasi stemmer
    stemmer = PorterStemmer()

    # Fungsi pra-pemrosesan teks
    def preprocess_text(text, lang_detected='en'):
        if lang_detected != 'en':
            return []
        text = text.lower()
        text = re.sub(r'[\W_0-9]+', ' ', text)
        words = word_tokenize(text)
        words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
        return words

    # Fungsi untuk mendapatkan rata-rata vektor Word2Vec untuk teks
    def get_avg_word2vec(text, model):
        vectors = [model.wv[word] for word in text if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    # Load the Word2Vec model using gensim's load method
    model = Word2Vec.load("word2vec_model.model")

    # Load the dataset
    df = pd.read_csv('coursera_dataset.csv')

    # Deteksi Bahasa untuk kolom yang relevan
    columns_to_detect = ['title', 'description', 'modules', 'instructor']

    def detect_language(text):
        try:
            if text.strip():
                return detect(text)
            else:
                return 'unknown'
        except LangDetectException:
            return 'unknown'

    for col in columns_to_detect:
        df[f'{col}_language_detected'] = df[col].apply(detect_language)

    # Filter hanya kursus berbahasa Inggris
    df = df[(df['title_language_detected'] == 'en') & 
            (df['description_language_detected'] == 'en') & 
            (df['modules_language_detected'] == 'en')]

    # Apply preprocessing for relevant columns
    columns_to_preprocess = ['title', 'description', 'modules', 'instructor']
    df['combined_clean'] = df.apply(lambda row: 
                                    preprocess_text(row['title'], row['title_language_detected']) + 
                                    preprocess_text(row['description'], row['description_language_detected']) + 
                                    preprocess_text(row['modules'], row['modules_language_detected']) + 
                                    preprocess_text(row['instructor'], row['instructor_language_detected']), axis=1)

    # Create document vectors
    df['doc_vector'] = df['combined_clean'].apply(lambda x: get_avg_word2vec(x, model))

    # Fungsi untuk mendapatkan rekomendasi berdasarkan input pengguna dengan threshold
    def get_user_recommendations(user_input, min_similarity=0.3):
        user_input_clean = preprocess_text(user_input, 'en')
        user_vector = get_avg_word2vec(user_input_clean, model)

        similarities = cosine_similarity([user_vector], np.vstack(df['doc_vector']))

        top_indices = [i for i in range(len(similarities[0])) if similarities[0][i] > min_similarity]

        keyword = user_input.lower()
        filtered_indices = [i for i in top_indices if 
                            keyword in df.iloc[i]['description'].lower() or 
                            keyword in df.iloc[i]['title'].lower() or 
                            keyword in df.iloc[i]['modules'].lower() or 
                            keyword in df.iloc[i]['instructor'].lower()]

        filtered_indices = sorted(filtered_indices, key=lambda i: similarities[0][i], reverse=True)[:9]  # Mengambil 9 rekomendasi teratas

        return df.iloc[filtered_indices]

    # Streamlit UI

    st.title('Course Recommendation System')
    st.write('Discover your new journey in learning. 🎓')

    # Tambahkan CSS untuk memperbesar komponen
    st.markdown("""
        <style>
        .course-container {
            text-align: center;
            margin-bottom: 30px;
        }
        .course-container img {
            max-width: 100%;
            height: auto;
        }
        .course-title {
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;
        }
        .expander {
            background-color: #f0f0f5;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .course-link {
            display: block;
            text-align: center;
            color: white;
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 5px;
            text-decoration: none;
            margin-top: 10px;
        }
        .course-link:hover {
            background-color: #45a049;
        }
        .block-container {
            max-width: 90%;
            padding: 1rem 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

    input_course = st.text_input("You can just describe the course you're looking for:", "")

    # Inisialisasi state untuk menyimpan hasil rekomendasi dan filter
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = pd.DataFrame()
    if 'filtered_recommendations' not in st.session_state:
        st.session_state.filtered_recommendations = pd.DataFrame()

    # Get recommendations based on user input
    if st.button('Check!'):
        if input_course:
            st.session_state.recommendations = get_user_recommendations(input_course)
            st.session_state.filtered_recommendations = st.session_state.recommendations
            st.write("Based on your input, we can recommend:")
        else:
            st.write("Please enter a course description or title.")

    # Display initial recommendations
    recommendations = st.session_state.filtered_recommendations
    if not recommendations.empty:
        cols = st.columns(3)  # Membuat layout grid dengan 3 kolom untuk tampilan yang lebih besar
        for index, row in recommendations.iterrows():
            with cols[index % 3]:  # Menampilkan kursus dalam kolom grid
                st.markdown(f"""
                    <div class="course-container">
                        <img src="{row['image']}" alt="{row['title']}">
                        <div class="course-title">{row['title']}</div>
                    </div>
                """, unsafe_allow_html=True)
                with st.expander("Description"):
                    st.write(row['description'])
                with st.expander("Instructor"):
                    st.write(row['instructor'])
                with st.expander("Modules"):
                    st.write(row['modules'])
                st.markdown(f"""
                    <a class="course-link" href="{row['url']}" target="_blank">Go to Course</a>
                """, unsafe_allow_html=True)

    # Filter box
    if not st.session_state.recommendations.empty:
        st.write("Filter your results:")
        with st.expander("Filter options", expanded=True):
            st.markdown('<div class="filter-box">', unsafe_allow_html=True)
            min_rating = st.slider('Choose a minimum rating:', 0.0, 5.0, 0.0, 0.1, key='rating')
            selected_level = st.selectbox('Filter by level:', ['All'] + df['level'].unique().tolist(), key='level')
            selected_price = st.selectbox('Filter by price:', ['All', 'Free', 'Enroll'], key='price')
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button('Apply Filter'):
                recommendations = st.session_state.recommendations.copy()
                if min_rating > 0:
                    recommendations = recommendations[recommendations['rating'] >= min_rating]
                if selected_level != 'All':
                    recommendations = recommendations[recommendations['level'] == selected_level]
                if selected_price != 'All':
                    if selected_price == 'Free':
                        recommendations = recommendations[recommendations['price'].str.lower() == 'free']
                    else:
                        recommendations = recommendations[recommendations['price'].str.lower() != 'free']

                st.session_state.filtered_recommendations = recommendations.head(9)  # Tampilkan top 9 setelah filter

                if not recommendations.empty:
                    st.write("Filtered results:")
                    cols = st.columns(3)  # Membuat layout grid dengan 3 kolom untuk tampilan yang lebih besar
                    for index, row in recommendations.head(9).iterrows():
                        with cols[index % 3]:  # Menampilkan kursus dalam kolom grid
                            st.markdown(f"""
                                <div class="course-container">
                                <div class="course-container">
                                    <img src="{row['image']}" alt="{row['title']}">
                                    <div class="course-title">{row['title']}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                                <a class="course-link" href="{row['url']}" target="_blank">Go to Course</a>
                            """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    run()
