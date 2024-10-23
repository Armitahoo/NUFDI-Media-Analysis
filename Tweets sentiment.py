import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import re

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Load the data from the original Excel file for sentiment analysis
file_path_sentiment = 'C:/Users/user/OneDrive/Documents/Advanced Willy/NUFDI/NUFDI/Translations_and_Sentiments (3) (1).xlsx'
df_sentiment = pd.read_excel(file_path_sentiment)

# Load the data from the new Excel file for hashtag count
file_path_hashtag = 'C:/Users/user/OneDrive/Documents/Advanced Willy/NUFDI/NUFDI/Tweets Master File.xlsx'
df_hashtag = pd.read_excel(file_path_hashtag)

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# List of hashtags to consider as negative
negative_hashtags = ['#election_circus', '#finger_in_blood', '#RezaPahlavi', 'circus', 'shot off internet', 
                     'turn off internet', 'blood', 'sheep', 'Crown Prince', 'Reza Pahlavi', 'javidshah', '#Sheep_census', 'Sanction_of_the_election_circus', 'boycott', '#Circus_election', 'brainless', 'Rainbow', '#Rainbow_god', 'IRGC', 'blacklist', 'Pahlavi era', 'take to the streets', 'ElectionCircus', '#No_to_the_election_circus', '#mahsa_amini', 'woman life freedom', 'woman_life_freedom', 'freedom', 'Mamad', 'corruption', 'King', 'Kings', 'cartel', 'cocaine',
                     '#remembering_the_crime', 'diarrhea', 'corrupt', '#مریم_رجوی', '#Marym_Rajavi', 'Majesty', 'visit Pyongyang', '#President_prover', '@PahlaviReza', '@AlinejadMasih', '#Toomaj_Saleh', '#Toomaj_Salehi', 'Petiareh', 'Azadi', '#free_life_woman', '#Zen_Zandagi_Azadi', '@IranIntl', '@ManotoNews', '@Aryammehr_2', '#رای_بی_رای', '#no_to_islamic_republic', 'Khomeinis criteria', '#Zeinab_Jalalian', '@pouriazeraati', 'Mir Salim', '#vote_by_vote', 'buy likes', 'buy legitimacy', '1500', 'Aban', 'clerics', 'leftist', 'wide butt', '@NickSotoudeh', 'clown', '#PS752', '#Rai_Man_Sarngoni', '#Javed', 'queers', 'regime', 'shame', 'royalists', '@ps752justice', 'puppet', 'actor', 'bloody', 'crackdown', '@HassanRonaghi', '@Yasnatoomaj', '@esmaeilion', 'apologist', 'victory', '#FuckIslamicRegime', '#FuckIslam', 'Mousavis', '#Saqez', 'cell', 'seven colored', 'snake', 'Putin', 'propagandist', 'propaganda', 'Construction', 'ISIS', 'Basiji', 'Basij', 'Sepah', '#free_life_woman', 'mercenery', 'killing patrol', 'beating girls', '#Baluchistan', 'Noor', 'Nour', 'Nur', 'Khomeini', 'Khamenei', 'mullah', '@iranwire', 'money', 'pot', 'mandatory hijab', 'not healed', 'rubbing', 'MEK', 'lie', 'lier', 'lies', '#Block_Governments', 'filth', '#IRGCterrorists', '#Sistan_and_Baluchistan', 'Salita_I am', 'women', 'garbage', 'thirsty', 'prostitute', 'Pasdaran', 'legitimacy', 'forgive', 'forget', 'motherfuckers', 'fundimetalism', 'ghorosnehh', 'snide', '#Death_to_the_totality_of_the_Islamic_Republic', '#death_to_khamenei', 'LGBTQ', 'light', '#noor_design']
positive_hashtags = ['radical', 'Jalili 2024', 'Resistence Axis', 'Dr. Jalili', 'Martyr Raisi', 'Zionist', 'Islamic Democracy', 'Long live Ghalibaf', 'haji', 'electing doctors', 'I am single', 'servent of the people', 'Dr Saeid', "#For_Iran", 'unity ', 'secret jew', 'neutralize the sanctions']

# Set to store seen tweets
seen_tweets = set()

# Dictionary to map variations of hashtags to a normalized form
hashtag_normalization_dict = {
    'election_circus': 'electioncircus',
    'Election_circus': 'electioncircus',
    'electioncircus': 'electioncircus',
    'ElectionCircus': 'electioncircus'
    # Add more variations as needed
}

# Function to normalize hashtags
def normalize_hashtag(hashtag):
    return hashtag_normalization_dict.get(hashtag.lower().replace('_', ''), hashtag)

# Function to calculate sentiment polarity
def get_sentiment(tweet):
    if isinstance(tweet, str):
        # Check if tweet has been seen before
        if tweet in seen_tweets:
            return None  # Skip sentiment analysis for duplicate tweet
        
        # Add tweet to seen set
        seen_tweets.add(tweet)
        
        # Check for negative hashtags
        for hashtag in negative_hashtags:
            if hashtag in tweet:
                return -1.0  # Return a negative sentiment score
        
        # Check for positive hashtags
        for hashtag in positive_hashtags:
            if hashtag in tweet:
                return 1.0  # Return a positive sentiment score
        
        # Perform regular sentiment analysis
        sentiment = sia.polarity_scores(tweet)
        return sentiment['compound']
    else:
        return None

# Apply the function to the 'Tweet' column
df_sentiment['Sentiment'] = df_sentiment['Tweet'].apply(get_sentiment)

# Filter out tweets that were skipped due to duplication
df_sentiment = df_sentiment.dropna()

# Save the results to a new Excel file
output_file_path = 'tweets_with_sentiment.xlsx'
df_sentiment.to_excel(output_file_path, index=False)

# Count the number of each sentiment type
negative_count = (df_sentiment['Sentiment'] < 0).sum()
positive_count = (df_sentiment['Sentiment'] > 0).sum()
neutral_count = (df_sentiment['Sentiment'] == 0).sum()

# Create a bar chart
sentiment_counts = [negative_count, positive_count, neutral_count]
sentiment_labels = ['Negative', 'Positive', 'Neutral']
colors = ['red', 'green', 'grey']

plt.figure(figsize=(8, 6))  # Set figure size (width, height) in inches
plt.bar(sentiment_labels, sentiment_counts, color=colors)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis Results')

# Save the bar chart as a PNG file with higher resolution (300 DPI)
bar_chart_output_path = 'sentiment_analysis_bar_chart.png'
plt.savefig(bar_chart_output_path, dpi=300)

# Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=sentiment_labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Analysis Distribution')

# Save the pie chart as a PNG file with higher resolution (300 DPI)
pie_chart_output_path = 'sentiment_analysis_pie_chart.png'
plt.savefig(pie_chart_output_path, dpi=300)

# Extract hashtags and count their occurrences with normalization from the new file
def extract_and_normalize_hashtags(tweet):
    if isinstance(tweet, str):
        hashtags = re.findall(r'#\w+', tweet)
        normalized_hashtags = [normalize_hashtag(tag) for tag in hashtags]
        return normalized_hashtags
    return []

all_hashtags = df_hashtag['Tweet'].apply(extract_and_normalize_hashtags).explode()
hashtag_counts = Counter(all_hashtags)

# Convert the hashtag counts to a DataFrame
hashtag_counts_df = pd.DataFrame(hashtag_counts.items(), columns=['Hashtag', 'Count']).sort_values(by='Count', ascending=False)

# Save the hashtag counts to a new Excel file
hashtag_counts_output_file_path = 'hashtag_counts.xlsx'
hashtag_counts_df.to_excel(hashtag_counts_output_file_path, index=False)

print(f"Sentiment analysis completed and saved to '{output_file_path}'")
print(f"Bar chart saved as '{bar_chart_output_path}'")
print(f"Pie chart saved as '{pie_chart_output_path}'")
print(f"Hashtag counts saved to '{hashtag_counts_output_file_path}'")
