# **GuideMe**


## **Problem Statement**

Students often face challenges in finding courses that align with their specific interests. The process of manually searching through extensive course catalogs can be time-consuming and overwhelming. Additionally, students may have difficulty identifying courses taught by particular instructors or covering specific topics. There is a need for an efficient and accurate solution to streamline this search process and connect students with their desired courses quickly.

GuideMe is a recommender system project designed to help students find their desired courses using input search. By leveraging Natural Language Processing (NLP) with Word2Vec methods, GuideMe aims to efficiently connect students with courses that best match their interests and preferences.

## **Project Objectives**
To provide accurate recommedations to users, GuideMe is optimized through the following steps:
- Analyze 4 text columns (title, description, instructors, modules) by employing NLP with Word2Vec methods.
- Measure similarity between text input and the dataset using Cosine Similarity to find the best match.
- Recommend courses to users with similarity score above 95%.
- Filtering Recommended course based on course's rating, level and price. 

## **Dataset**
The dataset is acquired from Kaggle [Coursera Courses Dataset](https://www.kaggle.com/datasets/muhammadanasmahmood/coursera-courses-dataset) and through web scraping the online learning platform, Coursera. Visit the [Coursera Homepage](https://coursera.org/) to explore more the available courses.


## Workflow
### Data Engineering
- Data acquisition: collected data by web scraping Coursera page.
- Data Cleaning: checked and manipulated data to create a more suitable dataset for data modelling.
### Data Science
- Model Development: Built a recommendation model using NLP with Word2Vec methods tu capture semantic meanings between words.
- Model Deployment: Deployed model using Streamlit.
### Data Analysis
- Visualization: Created Tableu dashboard to visualize key findings of the dataset. 
- Exploratory Data Analysis (EDA): Created PowerPoint presentation and EDA.


## **Deployment**
Hugging Face [GuideMe - Course Recommender System](https://huggingface.co/spaces/MARVT/GuideMe_CourseRecommenderSystem_FinalProject)


## **Authors**

- **Maurits Tobing**
  - Role: Data Engineer  
  - [Github](https://github.com/MARVT7) | [LinkedIn](https://www.linkedin.com/feed/)

- **Iznia Azyati**
  - Role: Data Scientist 
  - [Github](https://github.com/Iznia) | [LinkedIn](https://www.linkedin.com/in/iznia-azyati/)

- **Talitha Salsabila**
  - Role: Data Analyst
  - [Github](https://github.com/talithasalsa) | [LinkedIn](https://www.linkedin.com/in/talithasalsabila/)

