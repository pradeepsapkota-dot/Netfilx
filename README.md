# Netflix Data Analysis & Machine Learning Project

## Project Description

This project demonstrates a complete machine learning pipeline and data analysis project completed for the Auspify Data Science Internship. The goal of the project was to derive business insights from the Netflix dataset using a variety of machine learning algorithms along with time series forecasting.

The project has gone above and beyond the requirements of the internship by completing 5 different tasks to display skills in data engineering, prediction models, clustering and recommendation systems.

## Project Structure

The project has been divided into sections for each task, with a central loader script to keep the code clean and DRY.

- Dataset - This holds the raw data of the Netflix catalogue in CSV format.
- Images - This holds all the visualizations and outputs from the models.
- Task1_Recommender - This uses content based filtering (TF-IDF & Cosine Similarity) to recommend content.
- Task2_Content_Type_Prediction - This uses a Random Forest classifier to predict if a content is a Movie or TV Show.
- Task3_Audience_Rating_Classification - This uses a Decision Tree to classify audience demographics (Kids/Teens/Adults).
- Task4_Content_Segmentation - This uses K-Means clustering to segment movie runtimes and historical periods.
- Task5_Netflix_Trend_Forecasting_Model - This uses time series exponential smoothing to forecast monthly content additions.

- Loader.py - This is a central script used to load in the data, format dates and remove hidden characters.

## Highlights & Techniques

No data leakage! Features were removed that would give away the target (i.e. removing duration units when classifying TV or Movies).
Feature Engineering! Multiple different categorical features were combined to create tags, these were then vectorized.
Time Series Analysis! Complex time series aggregations were performed (.resample) to identify how the company has been growing over time, despite real world volatility (i.e. pandemic affecting production).
Unsupervised Learning! Features were scaled (StandardScaler) and clustered based on distance to mathematically demonstrate the shift in strategy towards modern short form content.

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
python task1_recommender.py
```
