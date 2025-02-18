# streaming_processor.py
import time
import random

def simulate_streaming_data():
    # Simulate streaming user interactions
    users = [1, 2, 3]
    items = [101, 102, 103]
    while True:
        user = random.choice(users)
        item = random.choice(items)
        rating = random.randint(1, 5)
        data_point = {"user_id": user, "item_id": item, "rating": rating}
        yield data_point
        time.sleep(2)  # simulate delay between data points

if __name__ == '__main__':
    stream = simulate_streaming_data()
    for _ in range(5):
        print(next(stream))
