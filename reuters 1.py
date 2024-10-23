
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = 'C:/Users/user/OneDrive/Documents/Advanced Willy/NUFDI/NUFDI/Reuters.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Ensure the 'Data' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove rows where 'Data' could not be converted to datetime
df = df.dropna(subset=['Date'])

# Count the number of articles published each month
df['YearMonth'] = df['Date'].dt.to_period('M')
article_counts = df.groupby('YearMonth').size().reset_index(name='article_count')

# Ensure 'YearMonth' is in string format for plotting
article_counts['YearMonth'] = article_counts['YearMonth'].astype(str)

# Plot the number of articles published each month
plt.figure(figsize=(12, 6), dpi=300)
sns.lineplot(x='YearMonth', y='article_count', data=article_counts, marker='o')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.title('Number of Articles Published Each Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('articles_per_month_WSJ.png')
plt.show()

# Ensure the 'Data' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove rows where 'Data' could not be converted to datetime
df = df.dropna(subset=['Date'])

# List of words to count
words_to_count = ['women', 'sanction', 'reform', 'democracy', 'right', 'nuclear']

# Initialize a dictionary to store word counts
word_counts = {word: 0 for word in words_to_count}

# Convert the article text to lowercase for case-insensitive counting
df['Text'] = df['Text'].str.lower()

# Count the occurrences of each word in the dataset
for word in words_to_count:
    word_counts[word] = df['Text'].str.count(word).sum()

# Convert the word_counts dictionary to a DataFrame for plotting
word_counts_df = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Count'])

# Plot the word counts
plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(x='Word', y='Count', data=word_counts_df)
plt.xlabel('Words')
plt.ylabel('Count')
plt.title('Word Frequency in Articles')
plt.tight_layout()
plt.savefig('word_frequency_WSJ.png')
plt.show()