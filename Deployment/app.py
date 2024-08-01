import eda
import recommender
import streamlit as st

# Add side bar for navigation
navigation = st.sidebar.selectbox('Navigation', ['Home', 'Exploratory Data Analysis', 'Course Recommendation'])

st.sidebar.markdown('# About')

# Introduction
st.sidebar.write('''This tool is designed to explore and recommend courses. It employs advanced data analysis and machine learning models to offer insights and recommendations that can assist users in understanding and selecting courses.''')

# Features
st.sidebar.write('''### Key Features:
- **Exploratory Data Analysis**: Analyze the data to uncover patterns and insights related to courses.
- **Course Recommendation**: Use predictive models to recommend relevant courses based on user input.''')

# Target Audience
st.sidebar.write('''### Who can benefit?
- **Learners**: Discover courses that match their interests and career goals.
- **Course Providers**: Analyze and improve course offerings based on data-driven insights.
- **Data Scientists**: Develop and evaluate machine learning models for course recommendation.''')

# Tools
st.sidebar.write('''### Tools Utilized:
- `Python`: For backend operations and model computations.
- `Streamlit`: For creating this interactive web application.
- `Gensim`: For implementing word embedding models.''')

# Define the Home Page
def show_home():
    # Create title and introduction
    st.title('Welcome to GuideMe')

    # Add image
    st.image('logo.png')
    st.write('')
    
    st.write('''This application is specifically designed to facilitate both exploratory data analysis and course recommendations. It provides users with advanced analytical tools that help in understanding trends and patterns within the data. To begin, please use the navigation menu on the left side of the screen to select the particular module that you intend to explore. Whether you're looking to uncover insights or receive course recommendations, this tool equips you with the necessary resources to effectively analyze and select the best courses available.''')
    st.write('''Use the navigation panel on the left to select the module you want to use.''')    
    st.markdown('---')

    
    # Dataset section
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image('gambar_dataset.png', use_column_width=True)
    with col2:
        st.markdown('### Dataset')
        st.markdown('''The dataset is obtained from a credible source [Coursera Courses Dataset](https://www.kaggle.com/datasets/muhammadanasmahmood/coursera-courses-dataset) and through web scraping the online learning platform, Coursera. Visit the [Coursera Homepage](https://coursera.org/) to explore more the available courses.''')
    
    # Problem Statement section
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.image('Problem_Statement.png', use_column_width=True)
    with col2:
        st.markdown('### Problem Statement')
        st.markdown('''Choosing the right course can greatly affect a learner's overall experience, including skill acquisition, career advancement, and personal development. With many online courses available, learners often **rely on these recommendations** to make informed decisions about which course to choose. However, the large number of courses can make it difficult and **time-consuming** to sift through and understand the best options available.''')
        st.markdown('''**CourseRecommender** aims to solve this problem by using advanced Natural Language Processing (NLP) techniques to analyze course details quickly and accurately. By processing and understanding a large number of courses, CourseRecommender can determine which courses are most relevant and beneficial for learners.''')
    
    # Objective section
    st.write('')
    st.write('')
    st.write('')
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.image('gambar_objective.png', use_column_width=True)

    with col2:
        st.markdown('### Objective')
        st.markdown('''The main goal of **CourseRecommender** is to improve the decision-making process for learners by providing personalized course recommendations based on the analysis of course data. Specifically, CourseRecommender aims to:

- Analyze course details to recommend the most relevant courses based on user input, with accuracy serving as the metric.
- Recommend courses with similar positive characteristics for users who have seen favorable reviews.
- Suggest top-rated alternative courses for users who are looking to expand their skill set or find better options for learning.''')


if navigation == 'Home':
    show_home()
elif navigation == 'Exploratory Data Analysis':
    eda.run()
elif navigation == 'Course Recommendation':
    recommender.run()
