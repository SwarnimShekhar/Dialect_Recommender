# collaborative_filtering.py
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

class CollaborativeFilteringRecommender:
    def __init__(self, ratings_file='data/ratings.csv'):
        self.ratings_file = ratings_file
        self.model = None
        self.trainset = None

    def load_data(self):
        df = pd.read_csv(self.ratings_file)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
        return data

    def train_model(self):
        data = self.load_data()
        trainset, _ = train_test_split(data, test_size=0.25, random_state=42)
        self.model = SVD(random_state=42)
        self.model.fit(trainset)
        self.trainset = trainset
        return self.model

    def get_recommendations(self, user_id, top_n=5):
        # Read the ratings to know which items exist and which the user already rated
        df = pd.read_csv(self.ratings_file)
        all_items = df['item_id'].unique()
        rated_items = df[df['user_id'] == user_id]['item_id'].tolist()
        recommendations = []
        for item in all_items:
            if item not in rated_items:
                pred = self.model.predict(user_id, item)
                recommendations.append((item, pred.est))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]

if __name__ == '__main__':
    rec_sys = CollaborativeFilteringRecommender()
    rec_sys.train_model()
    user_id = 1
    recs = rec_sys.get_recommendations(user_id)
    print(f"Recommendations for user {user_id}: {recs}")