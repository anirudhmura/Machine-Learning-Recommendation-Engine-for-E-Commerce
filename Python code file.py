# 1. Import necessary libraries
import pandas as pd
import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate

# 2. Read all relevant datasets
customers = pd.read_csv("olist_customers_dataset.csv")
geolocation = pd.read_csv("olist_geolocation_dataset.csv")
order_items = pd.read_csv("olist_order_items_dataset.csv")
order_payments = pd.read_csv("olist_order_payments_dataset.csv")
order_reviews = pd.read_csv("olist_order_reviews_dataset.csv")
orders = pd.read_csv("olist_orders_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
sellers = pd.read_csv("olist_sellers_dataset.csv")
product_category_name_translation = pd.read_csv("product_category_name_translation.csv")

# 3. Check counts of different order statuses
orders.order_status.value_counts()

# 4. Create a master dataframe containing details of the products, reviews and customers
product_ratings = order_reviews.merge(
    order_items, on='order_id', how='left').merge(
    orders, on='order_id', how='left').merge(
    customers, on='customer_id', how='left').merge(
    products, on='product_id', how='left').merge(
    product_category_name_translation, on='product_category_name', how='left')

# 5. Filter for valid order statuses
product_ratings_valid = product_ratings[product_ratings.order_status.isin(['delivered', 'shipped', 'invoiced'])]

# 6. Analyze product categories
products.merge(product_category_name_translation, on='product_category_name',
               how='left').groupby('product_category_name_english')['product_id'].count().sort_values(ascending=False)

product_ratings_valid.groupby('product_category_name_english')['order_id'].count().sort_values(ascending=False)

# 7. Check average number of products per order
average_products_per_order = product_ratings_valid.groupby('order_id')['product_id'].count().reset_index().sort_values(
    'product_id').product_id.mean()

# 8. Plot distribution of products per order
product_ratings_valid.groupby('order_id')['product_id'].count().reset_index().product_id.value_counts().plot(
    kind='bar', use_index=True)

# 9. Calculate re-purchase rate
n_repeat_customers = customers[customers.duplicated(subset='customer_unique_id', keep=False)].customer_unique_id.nunique()
total_customers = product_ratings_valid.customer_unique_id.nunique()
repeat_percent = n_repeat_customers/total_customers*100

# 10. Define collaborative filtering model function
def collab_filter_model(customer_id, product_id, n_recs, df, cosine_sim, consine_sim_T, svd_model, rmat_T, rmat_T_index):
    # Function implementation...

# 11. Build the scoring matrix for Item-based and User-based Collaborative Filtering Model
rmat = product_ratings_valid.groupby(['product_id', 'customer_unique_id'])['review_score'].max().unstack().fillna(0)
rmat_index = rmat.index
rmat_T_index = rmat.T.index
rmat_T_columns = rmat.T.columns
rmat_T = scipy.sparse.csr_matrix(rmat.T.values)
rmat = scipy.sparse.csr_matrix(rmat.values)

# 12. Compute cosine similarity matrices
cosine_sim = cosine_similarity(rmat)
cosine_sim_T = scipy.sparse.csr_matrix(cosine_similarity(rmat_T))

cosine_sim = pd.DataFrame(cosine_sim, index=rmat_index, columns=rmat_index)
cosine_sim_T = pd.DataFrame(cosine_sim_T, index=rmat_T_index, columns=rmat_T_index)

# 13. Prepare data for collaborative filtering model
reader = Reader()
data = Dataset.load_from_df(product_ratings_valid[['customer_unique_id', 'product_id', 'review_score']], reader)

# 14. Split data into train and test sets
trainset, testset = train_test_split(data, test_size=0.3)

# 15. Train SVD model
svd = SVD()
svd.fit(trainset)

# 16. Make predictions on test set
test_predictions = svd.test(testset)

# 17. Perform cross-validation
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# 18. Generate recommendations for a sample customer and product
c_id = product_ratings_valid['customer_unique_id'].values[0]
p_id = product_ratings_valid['product_id'].values[0]
n_recs = 5
item_recommendations = collab_filter_model(c_id, p_id, n_recs, product_ratings_valid,
                                           cosine_sim, cosine_sim_T, svd, rmat_T, rmat_T_index)

# 19. Format and display recommendations
new = item_recommendations[['product_id', 'product_category_name_english', 'est']]
new.rename(columns={'product_id': 'Product ID',
                    'product_category_name_english': 'Product Category',
                    'est': 'Estimated Score'}).reset_index(drop=True)