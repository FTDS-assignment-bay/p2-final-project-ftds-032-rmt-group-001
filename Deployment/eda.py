import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Create the main program
def run():
    
    st.title("Exploratory Data Analysis")

    # Load the dataset
    dataset_path = "coursera_dataset.csv"
    df = pd.read_csv(dataset_path)

    # Additional image
    st.image("gambar1.jpg")

    # Function to generate word cloud
    def generate_word_cloud(data, title):
        text = ' '.join(data['category'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_title(title, fontsize=14, pad=20)
        ax.axis("off")
        st.pyplot(fig)
        st.write("**Insight:** Courses in Coursera are dominated by IT and Data domains. "
                "The top 3 course categories are Cloud Computing, Data Analysis, Business Essentials.")

    # Function to plot course level distribution
    def plot_course_level_distribution(df):
        labels = df['level'].value_counts().index
        sizes = df['level'].value_counts().values
        colors = ['green', 'orange', 'yellow']
        explode = [0.1, 0, 0.1]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(sizes, colors=colors, labels=labels, shadow=True, startangle=90, autopct='%.2f%%', explode=explode)
        ax.set_title('Course Level', fontsize=20)
        st.pyplot(fig)
        st.write("**Insight:** Most of the courses are designed for beginners. Only a few institutions provide courses "
                "for advanced and intermediate levels.")

    # Function to plot top associations by course level
    def plot_top_associations(df):
        df['count'] = np.random.randint(1, 100, size=len(df))
        top_10_df = df.groupby('association').sum().sort_values(by='count', ascending=False).head(5).reset_index()
        top_10_associations = top_10_df['association'].tolist()
        filtered_df = df[df['association'].isin(top_10_associations)]
        association_level = filtered_df.groupby(['association', 'level']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        association_level.plot(kind='bar', stacked=True, ax=ax, color=['red', 'green', 'orange'])

        for container in ax.containers:
            ax.bar_label(container, label_type='center')

        ax.set_title('Associated Institution with Course Level', fontsize=16)
        ax.set_xlabel('Associated Institution', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.legend(title='Course level')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        st.write("**Insight:** Coursera collaborates with various institutions to provide courses. "
                "Google is the most frequent provider, followed by Google Cloud, IBM Skills Network, General Assembly, "
                "and the University of Colorado.")

    # Function to plot top 5 instructors
    def plot_top_instructors(df):
        df['count'] = np.random.randint(1, 100, size=len(df))
        top_5_instructors = df.groupby('instructor').sum().sort_values(by='count', ascending=False).head(5).reset_index()
        top_5_instructors = top_5_instructors['instructor'].tolist()
        instructor_df = df[df['instructor'].isin(top_5_instructors)]
        instructor_count = instructor_df.groupby(['instructor']).size()

        fig, ax = plt.subplots(figsize=(8, 5))
        instructor_count.plot(kind='bar', stacked=False, ax=ax)

        for container in ax.containers:
            ax.bar_label(container, label_type='center')

        ax.set_title('Top 5 Instructors Frequently Teaching', fontsize=16)
        ax.set_xlabel('Instructors', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        st.write("**Insight:** Google has the highest number of course instructors. Other notable instructors teaching "
                "five or more courses include Alex Cowan, Greg Williams, and Jeffrey R. Chasnov.")

    # Overview function
    def overview(df):
        st.write("## Overview of the Dataset")
        st.write("""
        The dataset contains information about various courses available on Coursera. 
        It includes details such as the course title, category, type, level, description, 
        price, rating, duration, language, prerequisites, syllabus, modules, instructor, 
        association, and more. The key highlights of the dataset are:
        - **Total Courses:** {}
        - **Unique Categories:** {}
        - **Unique Instructors:** {}
        - **Price Range:** {} to {}
        - **Rating Range:** {} to {}
        """.format(
            df.shape[0],
            df['category'].nunique(),
            df['instructor'].nunique(),
            df['price'].min(),
            df['price'].max(),
            df['rating'].min(),
            df['rating'].max()
        ))


    # Streamlit app
    st.write("Discover insights from the Coursera dataset.")

    # Display raw data
    if st.checkbox("Show raw data"):
        st.dataframe(df)

    # Select analysis
    analysis_type = st.selectbox("Select Analysis", ["Overview", "Distribution Plots", "Categorical Analysis", "Word Cloud"])

    if analysis_type == "Overview":
        overview(df)
    elif analysis_type == "Distribution Plots":
        st.subheader("Course Level Distribution")
        plot_course_level_distribution(df)
    elif analysis_type == "Categorical Analysis":
        st.subheader("Top Associations by Course Level")
        plot_top_associations(df)
        st.subheader("Top 5 Instructors")
        plot_top_instructors(df)
    elif analysis_type == "Word Cloud":
        st.subheader("Word Cloud for Course Categories")
        generate_word_cloud(df, 'All Course Categories')


# Run the app
if __name__ == '__main__':
    run()
