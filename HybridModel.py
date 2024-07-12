import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Load the dataset
cars_data = pd.read_csv(r"Dataset\cars_data_clean.csv")

# Select relevant features for recommendation
features = [
    'listed_price', 'myear', 'body', 'transmission', 'fuel', 'km',
    'Engine Type', 'Length', 'Width', 'Height', 'Max Power Delivered', 'Color' ,
    'model' , 'images'
]

cars_required_data = cars_data[features]

# Define numerical and categorical features
# numerical_features = ['listed_price', 'myear', 'km', 'Length', 'Width', 'Height', 'Max Power Delivered']
# categorical_features = ['body', 'transmission', 'fuel', 'Engine Type', 'Color']
numerical_features = ['listed_price', 'myear', 'Length',]
categorical_features = ['body', 'transmission', 'fuel', 'Color']

features_to_be_used = numerical_features + categorical_features

car_data = cars_data[features_to_be_used]

# Ensure there are no missing values and duplicates
car_data = car_data.dropna()
car_data = car_data.drop_duplicates()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Define default fill values for each column type
fill_values = {
    'listed_price': 0,
    'myear': 0,
    'body': 'Unknown',
    'transmission': 'Unknown',
    'fuel': 'Unknown',
    'km': 0,
    'Engine Type': 'Unknown',
    'Length': 0.0,
    'Width': 0.0,
    'Height': 0.0,
    'Max Power Delivered': 0,
    'Color': 'Unknown',
    'model': 'Unknown',
    'images': 'No Image'
}

# Function to get recommendations based on KNN (content-based)
def get_knn_recommendations(user_transformed, knn_model, n_neighbors=10):
    distances, indices = knn_model.kneighbors(user_transformed, n_neighbors=n_neighbors)
    return indices.flatten()

# Function to get recommendations based on collaborative filtering
def get_collaborative_recommendations(user_transformed_svd, X_svd, n_neighbors=10):
    distances = np.linalg.norm(X_svd - user_transformed_svd, axis=1)
    indices = np.argsort(distances)[:n_neighbors]
    return indices

# Function to get hybrid recommendations
def get_hybrid_recommendations(user_transformed, user_transformed_svd, X_svd, knn_model, n_neighbors=10):
    knn_indices = get_knn_recommendations(user_transformed, knn_model, n_neighbors)
    collab_indices = get_collaborative_recommendations(user_transformed_svd, X_svd, n_neighbors)
    combined_indices = np.unique(np.concatenate((knn_indices, collab_indices)))
    return combined_indices


# Preprocess user input
def preprocess_user_input(user_data):
    user_df = pd.DataFrame([user_data])
    user_transformed = preprocessor.transform(user_df)
    return user_transformed

def hybrid_model(user_data):
    car_filtered_data = car_data[(car_data['body'] == user_data['body']) & (car_data['fuel'] == user_data['fuel'])]
    cars_filtered_required_data = cars_required_data[(cars_required_data['body'] == user_data['body']) & (cars_required_data['fuel'] == user_data['fuel'])]

    if(cars_filtered_required_data.shape[0] <= 10):
        return cars_filtered_required_data

    X = preprocessor.fit_transform(car_filtered_data)

    # Dynamically set the number of components for SVD
    n_components = min(50, X.shape[1])

    # Fit the KNN model for content-based filtering
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(X)

    # Collaborative Filtering using Matrix Factorization
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X)

    user_transformed = preprocess_user_input(user_data)
    
    # Applying SVD transformation to the user input
    user_transformed_svd = svd.transform(user_transformed)
    
    hybrid_indices = get_hybrid_recommendations(user_transformed, user_transformed_svd, X_svd, knn, n_neighbors=10)

    recommended_cars = cars_filtered_required_data.iloc[hybrid_indices]
    
    # Fill NaN values for specified features
    recommended_cars.fillna(fill_values, inplace=True)

    return recommended_cars


