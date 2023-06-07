"""
Script to preprocess the data from Movie Lense dataset
movies.csv contains the following columns: movieId, title, genres
    example: 1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy

ratings.csv contains the following columns: userId, movieId, rating, timestamp
    example: 1,296,5.0,1147880044
tags.csv contains the following columns: userId, movieId, tag, timestamp
    example: 4,7569,so bad it's good,1573943455
"""
#%% Imports
import os
# import ipdb
import pickle as pkl
from collections import OrderedDict
import numpy as np

#%% Globals
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_movie_lense")
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
TAGS_CSV = os.path.join(DATA_DIR, "tags.csv")
train_split = 0.8

genre_to_id_dict = {
    'Action': 0,
    'Adventure': 1,
    'Animation': 2,
    'Children': 3,
    'Comedy': 4,
    'Crime': 5,
    'Documentary': 6,
    'Drama': 7,
    'Fantasy': 8,
    'Film-Noir': 9,
    'Horror': 10,
    'Musical': 11,
    'Mystery': 12,
    'Romance': 13,
    'Sci-Fi': 14,
    'Thriller': 15,
    'War': 16,
    'Western': 17,
    'IMAX': 18,
    '(no genres listed)': 19,
}


# %% Movies dict
def get_movies_dict(movies_csv):
    """
    Returns a dictionary of movie_id to movie info.
    """
    movies_dict = {}
    with open(movies_csv, "r") as f:
        f.readline()  # skip header
        for line in f:
            splitted_line = line.strip().split(",")
            # Handle commas in movie titles:
            movie_id, title, genres = splitted_line[0], ",".join(splitted_line[1:-1]), splitted_line[-1]
            genres = genres.split("|")
            try:
                genres = [genre_to_id_dict[genre] for genre in genres]
            except KeyError:
                print("KeyError: {}".format(genres))
                print("Error on line: {}".format(line))
                raise KeyError
            movie_id = int(movie_id)
            movies_dict[movie_id] = {"title": title, "genres": genres}
    return movies_dict

# Save movies_dict
movies_dict = get_movies_dict(MOVIES_CSV)
with open(os.path.join(DATA_DIR, "movies_dict.pkl"), "wb") as f:
    pkl.dump(movies_dict, f)


#%% Ratings list
def get_ratings_list(ratings_csv):
    """
    Returns a list of dicts of user_id, movie_id, rating.
    """
    ratings_list = []
    with open(ratings_csv, "r") as f:
        f.readline()  # skip header
        for line in f:
            user_id, movie_id, rating, timestamp = line.strip().split(",")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            timestamp = float(timestamp)
            ratings_list.append({"user_id": user_id, "movie_id": movie_id, "rating": rating, "timestamp": timestamp})
    return ratings_list

# Save ratings_list
ratings_list = get_ratings_list(RATINGS_CSV)
with open(os.path.join(DATA_DIR, "ratings_list.pkl"), "wb") as f:
    pkl.dump(ratings_list, f)
    
    
#%% Train and test splits
shuffled_ratings_list = ratings_list.copy()
np.random.shuffle(shuffled_ratings_list)
train_split_idx = int(len(shuffled_ratings_list) * train_split)
train_ratings_list = shuffled_ratings_list[:train_split_idx]
test_ratings_list = shuffled_ratings_list[train_split_idx:]

# Save train_ratings_list
with open(os.path.join(DATA_DIR, "train_ratings_list.pkl"), "wb") as f:
    pkl.dump(train_ratings_list, f)

# Save test_ratings_list
with open(os.path.join(DATA_DIR, "test_ratings_list.pkl"), "wb") as f:
    pkl.dump(test_ratings_list, f)
