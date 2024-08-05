# Machine-Learning-Recommendation-Engine-for-E-Commerce

## Overview

The script performs the following main tasks:

1. Data Preprocessing and Exploration
   - Loads multiple datasets related to Olist's e-commerce operations
   - Merges datasets to create a comprehensive view of products, orders, and customers
   - Analyzes order statuses, product categories, and customer behavior

2. Purchase Behavior Analysis
   - Calculates average number of products per order
   - Determines the re-purchase rate among customers

3. Recommendation Engine Implementation
   - Utilizes an Item-based Collaborative Filtering Model
   - Implements both item-based and user-based cosine similarity
   - Employs Single Value Decomposition (SVD) for matrix factorization

4. Model Training and Evaluation
   - Splits data into training and testing sets
   - Trains an SVD model on the dataset
   - Performs cross-validation to assess model performance

5. Recommendation Generation
   - Provides a function to generate product recommendations for a given customer

## Requirements

- Python 3.7+
- pandas
- numpy
- scipy
- scikit-learn
- surprise

## Usage

1. Ensure all required libraries are installed.
2. Place all Olist dataset CSV files in the same directory as the script.
3. Run the script in a Python environment.

## Data

The script uses multiple CSV files from the Olist dataset, including customer data, order details, product information, and reviews.

## Output

The script outputs:
- Analysis of product categories and order behaviors
- Model performance metrics (RMSE and MAE)
- Sample product recommendations for a given customer

## Note

This script is for educational and demonstration purposes. The effectiveness of the recommendations should be validated in a real-world scenario before implementation.
