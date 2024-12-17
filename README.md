# Instagram-Post-Impressions

## Project Overview

This project aims to develop a predictive model for estimating the Impressions (or visibility) of Instagram posts based on various engagement metrics. The dataset used in this project is available on Kaggle and contains data about Instagram posts, including features such as Likes, Saves, Profile Visits, Follows, and Hashtag Count. The goal is to build multiple machine learning models, including Multiple Linear Regression, K-Nearest Neighbors (KNN), Decision Trees, Random Forest, XGBoost, and Artificial Neural Networks (ANN), to accurately predict the number of Impressions a post will receive.

## Problem Statement

Instagram has become one of the most influential social media platforms globally, with over a billion active users sharing millions of posts daily. This project develops a predictive model to estimate the Impressions (visibility) of social media posts based on engagement metrics such as Likes, Saves, Profile Visits, Follows, and Hashtag Count. By utilizing multiple regression models and machine learning techniques—including Multiple Linear Regression, K-Nearest Neighbors (KNN), Decision Trees, Random Forest, XGBoost, and Artificial Neural Networks (ANN)—the project aims to identify which metrics contribute most significantly to Impressions and evaluate the performance of different models. The insights gained from these models will help optimize social media strategies by tailoring content to maximize reach and visibility, benefiting marketers, advertisers, and social media influencers.

## Features

- **Likes:** The number of likes a post receives.
- **Saves:** The number of saves a post gets.
- **Profile Visits:** The number of visits to the profile associated with the post.
- **Follows:** The number of follows generated from the post.
- **Hashtag Count:** The number of hashtags used in the post.

## Technologies Used

- Python
- Google Colab
- Scikit-learn (for machine learning models)
- Pandas (for data manipulation)
- Matplotlib & Seaborn (for visualization)
- XGBoost (for advanced boosting model)
- TensorFlow/Keras (for Artificial Neural Networks)

## Installation Instructions

1. Clone this repository to your local machine using:

   ```bash
   git clone  https://github.com/hj2342/Instagram-Post-Impressions.git   
   ```

2. Navigate to the project directory:

   ```bash
   cd instagram-impressions-prediction
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. You can now run the Jupyter notebook or Python scripts to begin your analysis.

## Usage

- Load the dataset into the Python script or Jupyter notebook.
- Preprocess the data (handle missing values, feature scaling, etc.).
- Split the data into training and testing sets.
- Train the models (Multiple Linear Regression, KNN, Decision Tree, Random Forest, XGBoost, and ANN) and evaluate their performance.
- Analyze feature importance to determine the key factors influencing Impressions.

## Model Evaluation

The models will be evaluated based on the following metrics:

- **R-squared**: To measure the proportion of variance explained by the model.
- **Mean Squared Error (MSE)**: To assess the accuracy of the model's predictions.
- **Feature Importance**: To identify which features have the most impact on the target variable (Impressions).

## Next Steps

- **Improve Model Generalization**: Tuning hyperparameters and preventing overfitting for better generalization.
- **Handle Outliers**: Implement techniques for detecting and handling outliers to improve model robustness.
- **Include Additional Features**: Incorporate other potential features like post timing and content type to refine the models.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements, bug fixes, or new ideas.

## Contact

For any questions or feedback, please feel free to reach out to:

**Name:** Hariharan Janardhanan  
**Email:** hj2342@nyu.edu
**LinkedIn:** hariharan-janardhanan-483ba51ab
