# Netflix Data Analysis & Machine Learning Pipeline

## Project Overview

This is an example of a complete machine learning pipeline and data analysis project which was undertaken as part of the Auspify Data Science Internship. The objective of the project was to gain business insights from the Netflix data set using a number of different machine learning algorithms.

The project went above and beyond the requirements of the internship. As a result, the project includes all 6 tasks. This shows a vast range of skills including data engineering, predictive models, clustering, recommendations and automated business intelligence.

## Project Structure

The project has been divided into sections for each task, with a central loader script to keep the code clean and DRY.

- Dataset - This holds the raw data of the Netflix catalogue in CSV format.
- Images - This holds all the visualizations and outputs from the models.
- Task1_Recommender - This uses content based filtering (TF-IDF & Cosine Similarity) to recommend content.
- Task2_Content_Type_Prediction - This uses a Random Forest classifier to predict if a content is a Movie or TV Show.
- Task3_Audience_Rating_Classification - This uses a Decision Tree to classify audience demographics (Kids/Teens/Adults).
- Task4_Content_Segmentation - This uses K-Means clustering to segment movie runtimes and historical periods.
- Task5_Netflix_Trend_Forecasting_Model - This uses time series exponential smoothing to forecast monthly content additions.
- Task6_Success_Analytics_Engine - contains the complete ML pipeline (Gradient Boosting & NLP) to provide automated business intelligence regarding global expansion.

- Loader.py - This is a central script used to load in the data, format dates and remove hidden characters.

## Key Points & Techniques

Data leakage was avoided by auditing the data and removing any columns that provided artificial insight into the target. For example, the duration unit was dropped before predicting if it was a TV show or movie.
Feature Engineering was conducted to merge different categories (Directors, Genres, Countries) into one column with tags. This then was vectoriser using TF-IDF.
Time series analysis was conducted to show how the platform grew over time using `.resample`. This allowed us to see how the platform grew over time despite the real world issues such as delays due to the pandemic.
The data was scaled using `StandardScaler` and clustered together using the distance between the features. This helped to prove that the business have made the right decision to focus on short form modern content.
Automated business intelligence was created by building a modelops pipeline to compare the different algorithms. We also extracted the feature importance from the winning model so we can understand what factors are impacting the outcome.

## Technologies Used

Programming Language: Python
Data Handling: Pandas, NumPy
Machine Learning: Scikit-Learn (Random Forest, Decision Tree, K-Means, TF-IDF Vectoriser)
Time Series Forecasting: Statsmodels (Holt-Winters Exponential Smoothing)
Visualization: Matplotlib, Seaborn

## How to Run

**Clone the repository:**  
[View my Netflix Repository](https://github.com/pradeepsapkota-dot/Netfilx.git)

```bash
git clone https://github.com/pradeepsapkota-dot/Netfilx.git
cd Netflix
```

### Install dependencies:

Make sure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

### Run Tasks:

Each task has been placed into its own folder and uses the central `Loader.py` script. Navigate to the relevant folder and run the script in your terminal.

```bash
cd Task1_Recommender
python task1.py
```
