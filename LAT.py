import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import re
import seaborn as sns
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure the stopwords and VADER lexicon are downloaded
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load the Excel file
file_path = 'C:/Users/user/OneDrive/Documents/Advanced Willy/NUFDI/NUFDI/LATimes.xlsx'
df = pd.read_excel(file_path)

# Convert the text column to a single string
text = ' '.join(df['Text'])

# Define a function to replace specific words
def replace_words(text, replacements):
    for old_word, new_word in replacements.items():
        text = re.sub(r'\b' + old_word + r'\b', new_word, text)
    return text

# Define a function to clean the text
def clean_text(text, additional_stopwords):
    # Convert to lower case
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english')).union(set(additional_stopwords))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

replacements = {'protest': 'protest', 'protester': 'protest', "protesters": 'protest', 'protests': 'protest', 
                'sanction': 'sanction', 'sanctions': 'sanction', 'economy': 'sanction', 'economic': 'sanction', 'reform':'reform', 'reformist': 'reform', 'reformists':'reform'}

# Define additional words to remove
additional_stopwords = ['iran', 'iranian', 'said', 'one', 'irans', 'include', 'ms', 'many', 'according', 'day', 
                        'year', 'time', 'back', 'told', 'say', 'long', 'video', 'called', 'iranians', 'year', 
                        'years', 'seeing', 'saying', 'whose', 'see', 'make', 'include', 'including', 'saturday', 
                        'us', 'way', 'still', 'sunday', 'among', 'come', 'though', 'photo', 'wrote', 'monday', 
                        'want', 'last', 'others', 'every', 'take', 'going', 'week', 'added', 'friday', 'thousand', 
                        'made', 'place', 'hundred', 'may', 'seen', 'mr', 'videos', 'began', 'wednesday', 'night', 
                        'inside', 'days', 'trying', 'three', 'become', 'last', 'even', 'used', 'work', 'call', 
                        'several', 'showed', 'outside', 'around', 'thing', 'far', 'today', 'recent', 'well', 
                        'used', 'past', 'started', 'new', 'dont', 'nights', 'continue', 'thousands', 'city', 
                        'weeks', 'early', 'first', 'four', 'thursday', 'know', 'part', 'must', 'took', 'use', 
                        'taken', 'despite', 'country', 'tuesday', 'reuters', 'oct', 'dubai']

# Apply the replacements
text = replace_words(text, replacements)

# Clean the text
cleaned_text = clean_text(text, additional_stopwords)

# Generate the word cloud with higher resolution
wordcloud_obj = WordCloud(width=1600, height=800, background_color='white').generate(cleaned_text)

# Plot the word cloud with higher resolution
plt.figure(figsize=(20, 10), dpi=300)  # Increase the figure size for higher resolution
plt.imshow(wordcloud_obj, interpolation='bilinear')
plt.axis('off')
plt.show()

# Save the word cloud image with higher resolution
wordcloud_obj.to_file('wordcloud_high_res1_LA.png')

# Get the word frequencies
word_frequencies = Counter(cleaned_text.split())

# Get the top 10 most common words
top_10_words = word_frequencies.most_common(10)

# Separate the words and their frequencies for plotting
words, frequencies = zip(*top_10_words)

# Print specific word counts
specific_words = ['sanction', 'democracy', 'movement', 'reform', 'government', 'republic', 'antigovernment', 'antiregime', 'nuclear', 'sanction', 'tehran', 'kurdish']
for word in specific_words:
    print(f"The word '{word}' appears {word_frequencies[word]} times.")
    
# Plot the top 10 words
sns.despine()
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6), dpi=300)
plt.bar(words, frequencies, color='lightgreen')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words')
plt.savefig('top_10_words1.png')  # Save the figure as an image file
plt.show()

# Define a function to check co-occurrence within a window
def check_cooccurrence(text, word1, word2, window_size):
    words = text.split()
    cooccurrence_count = 0
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        if word1 in window and word2 in window:
            cooccurrence_count += 1
    return cooccurrence_count

# Check the co-occurrence of specific words within a window of 10 words
cooccurrence_pairs = [
    ('protest', 'democracy'),
    ('protest', 'israel'),
    ('protest', 'reform'),
    ('protest', 'right'),
    ('protest', 'sanction'),
    ('protest', 'women'),
    ('protest', 'kurdish')
]

window_size = 10

# Calculate the co-occurrence counts
cooccurrence_counts = []
for word1, word2 in cooccurrence_pairs:
    cooccurrence_count = check_cooccurrence(cleaned_text, word1, word2, window_size)
    cooccurrence_counts.append((word1, word2, cooccurrence_count))
    print(f"The words '{word1}' and '{word2}' co-occur {cooccurrence_count} times within a window of {window_size} words.")

# Create a bar chart for the co-occurrence counts
labels = [f'{pair[0]} & {pair[1]}' for pair in cooccurrence_pairs]
counts = [count for _, _, count in cooccurrence_counts]

sns.despine()
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6), dpi=300)
plt.bar(labels, counts, color='#ff4f33')
plt.xlabel('Word Pairs')
plt.ylabel('Co-occurrence Count')
plt.title('Co-occurrence of Word Pairs within 10-Word Window')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cooccurrence_counts1_LA.png')
plt.show()

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()

# Calculate sentiment for each row in the DataFrame
df['sentiment'] = df['Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Plot sentiment distribution
plt.figure(figsize=(12, 6), dpi=300)
sns.histplot(df['sentiment'], bins=20, kde=True)
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis of Text Data')
plt.savefig('sentiment_distribution_LA.png')
plt.show()

# Sentiment analysis for specific word
specific_word = 'nuclear'

# Filter sentences containing the specific word
filtered_df = df[df['Text'].str.contains(r'\b' + specific_word + r'\b', case=False, na=False)]

# Calculate sentiment for these filtered sentences
filtered_df['sentiment'] = filtered_df['Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Save the filtered sentences and their sentiment scores to a CSV file
filtered_df.to_csv(f'filtered_sentences_{specific_word}.csv', index=False)

# Plot sentiment distribution for the specific word
plt.figure(figsize=(12, 6), dpi=300)
sns.histplot(filtered_df['sentiment'], bins=20, kde=True)
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title(f'Sentiment Analysis of Texts Containing the Word "{specific_word}"')
plt.savefig(f'sentiment_distribution_{specific_word}_LA.png')
plt.show()

# Define the words to compare
words_to_compare = [
    (specific_words[0], specific_words[1]),  # sanction vs. demand
    (specific_words[2], specific_words[3]),  # death vs. killed
    (specific_words[4], specific_words[5]),  # government vs. regime
    (specific_words[6], specific_words[7]),   # antigovernment vs. antiregime
    (specific_words[8], specific_words[9]),
    (specific_words[10], specific_words[11])
]

# Get the counts for the specified words
word_counts = {word: word_frequencies[word] for pair in words_to_compare for word in pair}

# Create pie charts for the pairs
for pair in words_to_compare:
    labels = [pair[0], pair[1]]
    sizes = [word_counts[pair[0]], word_counts[pair[1]]]
    
    plt.figure(figsize=(8, 8), dpi=300)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'green'])
    plt.title(f'Comparison of {pair[0]} and {pair[1]}', bbox={'facecolor': 'w', 'edgecolor': 'w', 'pad': 10})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    plt.savefig(f'{pair[0]}_vs_{pair[1]}_LA.png')  # Save each pie chart as an image file
    plt.show()

# Sort the DataFrame by date
df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is in datetime format
df = df.sort_values(by='Date')

# Plot sentiment over time using a violin plot
plt.figure(figsize=(12, 6), dpi=300)
sns.violinplot(x=df['Date'].dt.to_period('M').astype(str), y=df['sentiment'])
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Score Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sentiment_over_time_violin_LA.png')
plt.show()

# Create a new column to count mentions of state-owned media
state_media = ['fars', 'tasnim', 'hrana', 'irna', 'isna']
df['state_media_mentions'] = df['Text'].apply(lambda x: sum([x.lower().count(media) for media in state_media]))

# Count the number of articles published each month
df['YearMonth'] = df['Date'].dt.to_period('M')
article_counts = df.groupby('YearMonth').size().reset_index(name='article_count')

# Aggregate the counts of state media mentions by month
media_mentions_over_time = df.groupby('YearMonth')['state_media_mentions'].sum().reset_index()

# Merge the two DataFrames to calculate the ratio
media_mentions_over_time = media_mentions_over_time.merge(article_counts, on='YearMonth')
media_mentions_over_time['mention_ratio'] = media_mentions_over_time['state_media_mentions'] / media_mentions_over_time['article_count']

# Convert the YearMonth to a timestamp for plotting
media_mentions_over_time['Date'] = media_mentions_over_time['YearMonth'].dt.to_timestamp()

# Plot the ratio of state media mentions to the number of articles published each month
plt.figure(figsize=(12, 6), dpi=300)
sns.lineplot(x='Date', y='mention_ratio', data=media_mentions_over_time, marker='o')
plt.xlabel('Date')
plt.ylabel('Mentions to Articles Ratio')
plt.title('Ratio of State-Owned Media Mentions to Articles Published Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('state_media_mentions_ratio_over_time_LA.png')
plt.show()
