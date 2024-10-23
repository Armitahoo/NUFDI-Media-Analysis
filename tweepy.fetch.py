import tweepy
import pandas as pd
import time

# Replace these with your own credentials
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAMuCuQEAAAAA1OcLNKk0QH%2FBl2rL2X3Xo6fsG8c%3DUU38G77Q8nJ0pwKu5OQKu6GtrobYkD2sdixq9lkA84JjRJBj10'

# Authenticate to Twitter
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

# Define the search query and fetch tweets
query = "انتحابات"
max_results = 100
limit = 1000  # Total number of tweets to fetch

# Store the tweet data and track unique tweet IDs
tweet_data = []
tweet_ids = set()
next_token = None
total_fetched = 0

while total_fetched < limit:
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=['author_id', 'created_at', 'id', 'text'],
            next_token=next_token
        )
        
        if not response.data:
            break

        for tweet in response.data:
            if tweet.id not in tweet_ids:
                tweet_ids.add(tweet.id)
                tweet_data.append({
                    'User ID': tweet.author_id,
                    'Created At': tweet.created_at,
                    'Tweet': tweet.text
                })

        total_fetched += len(response.data)
        
        if 'next_token' in response.meta:
            next_token = response.meta['next_token']
        else:
            break

    except tweepy.TooManyRequests:
        print("Rate limit reached. Sleeping for 15 minutes.")
        time.sleep(15 * 60)
        continue

# Create a DataFrame and save to CSV with UTF-8 encoding
df = pd.DataFrame(tweet_data)
df.to_csv('tweets9.csv', index=False, encoding='utf-8-sig')

print("Tweets have been saved to tweets.csv")
