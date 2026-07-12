# ============================================================
# Movie Recommendation System using K-Nearest Neighbors (KNN)
# ============================================================

# ============================================================
# Import Required Libraries
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sns.set_style("whitegrid")

# ============================================================
# Load Ratings Dataset
# ============================================================

ratings_columns = [

    "user_id",

    "movie_id",

    "rating",

    "timestamp"

]

ratings = pd.read_csv(

    "u.data",

    sep="\t",

    names=ratings_columns

)

# ============================================================
# Load Movies Dataset
# ============================================================

movies_columns = [

    "movie_id",

    "title",

    "release_date",

    "video_release_date",

    "IMDb_URL",

    "unknown",

    "Action",

    "Adventure",

    "Animation",

    "Children",

    "Comedy",

    "Crime",

    "Documentary",

    "Drama",

    "Fantasy",

    "Film_Noir",

    "Horror",

    "Musical",

    "Mystery",

    "Romance",

    "Sci_Fi",

    "Thriller",

    "War",

    "Western"

]

movies = pd.read_csv(

    "u.item",

    sep="|",

    names=movies_columns,

    encoding="latin-1"

)

# ============================================================
# Explore Ratings Dataset
# ============================================================

print("="*60)
print("Ratings Dataset")
print("="*60)

print(ratings.head())

print("\n")

print(ratings.info())

print("\n")

print(ratings.shape)

# ============================================================
# Explore Movies Dataset
# ============================================================

print("\n")

print("="*60)
print("Movies Dataset")
print("="*60)

print(movies.head())

print("\n")

print(movies.info())

print("\n")

print(movies.shape)

# ============================================================
# Missing Values
# ============================================================

print("\n")

print("="*60)
print("Missing Values")
print("="*60)

print(ratings.isnull().sum())

print()

print(movies.isnull().sum())

# ============================================================
# Duplicate Records
# ============================================================

print("\n")

print("="*60)
print("Duplicate Records")
print("="*60)

print("Ratings :", ratings.duplicated().sum())

print("Movies  :", movies.duplicated().sum())

# ============================================================
# Merge Datasets
# ============================================================

movie_data = pd.merge(

    ratings,

    movies,

    on="movie_id"

)

print("\n")

print("="*60)
print("Merged Dataset")
print("="*60)

print(movie_data.head())

print()

print(movie_data.shape)

# ============================================================
# Dataset Information
# ============================================================

print("\n")

print(movie_data.info())

# ============================================================
# Movie Rating Statistics
# ============================================================

movie_stats = movie_data.groupby(

    "title"

)["rating"].agg(

    ["mean","count"]

)

movie_stats.rename(

    columns={

        "mean":"Average Rating",

        "count":"Number of Ratings"

    },

    inplace=True

)

print("\n")

print("="*60)
print("Movie Statistics")
print("="*60)

print(movie_stats.head())

# ============================================================
# Top Rated Movies
# ============================================================

print("\n")

print("="*60)
print("Highest Rated Movies")
print("="*60)

print(

    movie_stats.sort_values(

        "Average Rating",

        ascending=False

    ).head(20)

)

# ============================================================
# Most Popular Movies
# ============================================================

print("\n")

print("="*60)
print("Most Rated Movies")
print("="*60)

print(

    movie_stats.sort_values(

        "Number of Ratings",

        ascending=False

    ).head(20)

)

# ============================================================
# Rating Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    ratings["rating"],

    bins=5,

    kde=False

)

plt.title("Rating Distribution")

plt.xlabel("Ratings")

plt.show()

# ============================================================
# Average Rating Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    movie_stats["Average Rating"],

    bins=30,

    kde=True

)

plt.title("Average Movie Rating")

plt.show()

# ============================================================
# Number of Ratings Distribution
# ============================================================

plt.figure(figsize=(8,5))

sns.histplot(

    movie_stats["Number of Ratings"],

    bins=40

)

plt.title("Number of Ratings")

plt.show()

# ============================================================
# Average Rating vs Rating Count
# ============================================================

plt.figure(figsize=(8,6))

sns.scatterplot(

    data=movie_stats,

    x="Number of Ratings",

    y="Average Rating"

)

plt.title("Average Rating vs Number of Ratings")

plt.show()

# ============================================================
# User Statistics
# ============================================================

user_stats = ratings.groupby(

    "user_id"

)["rating"].count()

print("\n")

print("="*60)
print("User Rating Statistics")
print("="*60)

print(user_stats.describe())

# ============================================================
# Create User-Movie Matrix
# ============================================================

movie_matrix = movie_data.pivot_table(

    index="title",

    columns="user_id",

    values="rating"

)

print("\n")

print("="*60)
print("User-Movie Matrix")
print("="*60)

print(movie_matrix.head())

print()

print(movie_matrix.shape)

# ============================================================
# Sparsity
# ============================================================

total_cells = movie_matrix.shape[0] * movie_matrix.shape[1]

filled_cells = movie_matrix.count().sum()

sparsity = (

    1 -

    (filled_cells / total_cells)

) * 100

print("\n")

print("="*60)
print("Matrix Sparsity")
print("="*60)

print(f"{sparsity:.2f}%")

print("\n")

# ============================================================
# Fill Missing Values
# ============================================================

# Replace missing ratings with 0.
# Here, 0 means "movie not rated by the user."

movie_matrix_filled = movie_matrix.fillna(0)

print("="*70)
print("Missing Values After Filling")
print("="*70)

print(movie_matrix_filled.isnull().sum().sum())

# ============================================================
# Convert to Sparse Matrix
# ============================================================

from scipy.sparse import csr_matrix

# Convert dense matrix into sparse matrix
# Saves memory and speeds up computation.

movie_sparse_matrix = csr_matrix(movie_matrix_filled.values)

print("\nSparse Matrix Created Successfully!")

# ============================================================
# Train KNN Model
# ============================================================

from sklearn.neighbors import NearestNeighbors

# Create KNN model

knn_model = NearestNeighbors(

    metric="cosine",

    algorithm="brute",

    n_neighbors=6

)

# Train the model

knn_model.fit(movie_sparse_matrix)

print("\nKNN Model Trained Successfully!")

# ============================================================
# Check Movie Exists
# ============================================================

movie_name = "Toy Story (1995)"

if movie_name in movie_matrix_filled.index:

    print(f"\nMovie Found : {movie_name}")

else:

    print("Movie Not Found!")

# ============================================================
# Find Similar Movies
# ============================================================

movie_index = movie_matrix_filled.index.get_loc(movie_name)

distances, indices = knn_model.kneighbors(

    movie_matrix_filled.iloc[movie_index, :].values.reshape(1, -1),

    n_neighbors=6

)

print("\n")

print("="*70)
print("Recommended Movies")
print("="*70)

for i in range(1, len(distances.flatten())):

    print(

        f"{i}. "

        f"{movie_matrix_filled.index[indices.flatten()[i]]}"

        f"  | Similarity Distance : "

        f"{distances.flatten()[i]:.4f}"

    )

# ============================================================
# Recommendation Function
# ============================================================

def recommend_movies(movie_name, n_recommendations=5):

    """
    Recommend similar movies using KNN.
    """

    if movie_name not in movie_matrix_filled.index:

        print("Movie not found in dataset.")

        return

    movie_index = movie_matrix_filled.index.get_loc(movie_name)

    distances, indices = knn_model.kneighbors(

        movie_matrix_filled.iloc[movie_index, :].values.reshape(1, -1),

        n_neighbors=n_recommendations + 1

    )

    print("\n")

    print("="*70)
    print(f"Movies Similar to '{movie_name}'")
    print("="*70)

    for i in range(1, len(indices.flatten())):

        print(

            f"{i}. "

            f"{movie_matrix_filled.index[indices.flatten()[i]]}"

            f" (Distance : {distances.flatten()[i]:.4f})"

        )

# ============================================================
# Test Recommendation Function
# ============================================================

recommend_movies(

    "Toy Story (1995)"

)

# ============================================================
# Recommend Another Movie
# ============================================================

recommend_movies(

    "Star Wars (1977)"

)

# ============================================================
# Display Model Information
# ============================================================

print("\n")

print("="*70)
print("KNN Model Information")
print("="*70)

print(f"Total Movies : {movie_matrix_filled.shape[0]}")

print(f"Total Users  : {movie_matrix_filled.shape[1]}")

print(f"Neighbors    : {knn_model.n_neighbors}")

print(f"Metric       : {knn_model.metric}")

print(f"Algorithm    : {knn_model.algorithm}")

print("\n")



# ============================================================
# Install fuzzywuzzy (Run only once if required)
# ============================================================

# pip install fuzzywuzzy
# pip install python-Levenshtein

# ============================================================
# Import Required Libraries
# ============================================================

from fuzzywuzzy import process

# ============================================================
# Fuzzy Movie Search
# ============================================================

# This helps users search even if they make spelling mistakes.

def find_movie(movie_name):

    movie_list = movie_matrix_filled.index.tolist()

    closest_match = process.extractOne(

        movie_name,

        movie_list

    )

    return closest_match[0]

# ============================================================
# Recommendation Function
# ============================================================

def recommend_movie(movie_name, n_recommendations=10):

    # Find closest matching movie

    movie_name = find_movie(movie_name)

    print("\n")

    print("="*70)
    print(f"Selected Movie : {movie_name}")
    print("="*70)

    movie_index = movie_matrix_filled.index.get_loc(

        movie_name

    )

    distances, indices = knn_model.kneighbors(

        movie_matrix_filled.iloc[movie_index, :].values.reshape(1,-1),

        n_neighbors=n_recommendations+1

    )

    recommendations = []

    print("\nTop Recommended Movies\n")

    for i in range(1,len(indices.flatten())):

        movie = movie_matrix_filled.index[

            indices.flatten()[i]

        ]

        similarity = 1 - distances.flatten()[i]

        recommendations.append(

            [movie, similarity]

        )

    recommendation_df = pd.DataFrame(

        recommendations,

        columns=[

            "Movie",

            "Similarity Score"

        ]

    )

    recommendation_df["Similarity Score"] = recommendation_df[

        "Similarity Score"

    ].round(4)

    print(recommendation_df)

    return recommendation_df

# ============================================================
# Test Recommendation System
# ============================================================

recommend_movie(

    "Titanic"

)

recommend_movie(

    "Toy Story"

)

recommend_movie(

    "Star Wars"

)

recommend_movie(

    "Batman"

)

# ============================================================
# Save KNN Model
# ============================================================

from joblib import dump

dump(

    knn_model,

    "movie_knn_model.joblib"

)

print("\nKNN Model Saved Successfully!")

# ============================================================
# Save Movie Matrix
# ============================================================

dump(

    movie_matrix_filled,

    "movie_matrix.joblib"

)

print("Movie Matrix Saved Successfully!")

# ============================================================
# Load Saved Model
# ============================================================

from joblib import load

loaded_knn = load(

    "movie_knn_model.joblib"

)

loaded_movie_matrix = load(

    "movie_matrix.joblib"

)

print("\nSaved Model Loaded Successfully!")

# ============================================================
# Recommendation Using Loaded Model
# ============================================================

movie = "Toy Story"

movie = find_movie(movie)

movie_index = loaded_movie_matrix.index.get_loc(movie)

distances, indices = loaded_knn.kneighbors(

    loaded_movie_matrix.iloc[movie_index,:].values.reshape(1,-1),

    n_neighbors=6

)

print("\n")

print("="*70)
print("Recommendations Using Loaded Model")
print("="*70)

for i in range(1,len(indices.flatten())):

    print(

        loaded_movie_matrix.index[

            indices.flatten()[i]

        ]

    )

# ============================================================
# Project Summary
# ============================================================

print("\n")

print("="*70)
print("Movie Recommendation System using KNN")
print("="*70)

print(f"Total Users           : {ratings['user_id'].nunique()}")

print(f"Total Movies          : {ratings['movie_id'].nunique()}")

print(f"Total Ratings         : {len(ratings)}")

print(f"Movie Matrix Shape    : {movie_matrix_filled.shape}")

print(f"Sparsity              : {sparsity:.2f}%")

print(f"KNN Metric            : {knn_model.metric}")

print(f"KNN Algorithm         : {knn_model.algorithm}")

print(f"Neighbors             : {knn_model.n_neighbors}")

print("\nRecommendation Engine Ready!")

print("="*70)

print("\nProject Completed Successfully!")
