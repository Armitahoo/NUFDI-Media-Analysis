import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import re
import seaborn as sns
from collections import Counter

# Ensure the stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the Excel file
file_path = 'C:/Users/user/OneDrive/Documents/Advanced Willy/NUFDI/NUFDI/sorted_sentiment_analysis.xlsx'  # Replace with your actual file path
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
                'sanction': 'sanction', 'sanctions': 'sanction', 'economy': 'sanction', 'economic': 'sanction'}

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
                        'taken', 'despite', 'country', 'tuesday']

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
wordcloud_obj.to_file('wordcloud_high_res.png')  # Save to a file with higher resolution

# Get the word frequencies
word_frequencies = Counter(cleaned_text.split())

# Get the top 10 most common words
top_10_words = word_frequencies.most_common(10)

# Separate the words and their frequencies for plotting
words, frequencies = zip(*top_10_words)

# Print specific word counts
specific_words = ['sanction', 'demand', 'death', 'killed', 'government', 'regime', 'antigovernment', 'antiregime']
for word in specific_words:
    print(f"The word '{word}' appears {word_frequencies[word]} times.")

# Plot the top 10 words
sns.despine()
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6), dpi=300)
plt.bar(words, frequencies, color='darkgreen')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words')
plt.savefig('top_10_words.png')  # Save the figure as an image file
plt.show()

# Define the words to compare
words_to_compare = [
    (specific_words[0], specific_words[1]),  # sanction vs. demand
    (specific_words[2], specific_words[3]),  # death vs. killed
    (specific_words[4], specific_words[5]),  # government vs. regime
    (specific_words[6], specific_words[7])   # antigovernment vs. antiregime
]

# Get the counts for the specified words
word_counts = {word: word_frequencies[word] for pair in words_to_compare for word in pair}

# Create pie charts for the pairs
for pair in words_to_compare:
    labels = [pair[0], pair[1]]
    sizes = [word_counts[pair[0]], word_counts[pair[1]]]
    
    plt.figure(figsize=(8, 8), dpi=300)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'pink'])
    plt.title(f'Comparison of {pair[0]} and {pair[1]}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(f'{pair[0]}_vs_{pair[1]}.png')  # Save each pie chart as an image file
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
    ('protest', 'right'),
    ('protest', 'democracy'),
    ('protest', 'sanction'),
    ('protest', 'women'),
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
plt.bar(labels, counts, color='#FFB533')
plt.xlabel('Word Pairs')
plt.ylabel('Co-occurrence Count')
plt.title('Co-occurrence of Word Pairs within 10-Word Window')
plt.savefig('cooccurrence_counts.png')
plt.show()
