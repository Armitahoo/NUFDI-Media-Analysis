# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:57:10 2024

@author: user
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Load the Excel file
file_path = "C:/Users/user/OneDrive/Documents/Advanced Willy/Final Data/Presentation/solaregypt/Table of Content (Just NYT).xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Ensure the date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Filter the data between the specified dates
start_date = pd.to_datetime('2022-09-16')
end_date = pd.to_datetime('2022-12-06')
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
filtered_df = df.loc[mask]

# Sort the dataframe by date
filtered_df = filtered_df.sort_values(by='Date')

# Initialize the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores
def get_sentiment(Text):
    sentiment_scores = sid.polarity_scores(Text)
    return sentiment_scores['compound']  # You can choose other scores like 'pos', 'neg', 'neu' if needed

# Apply the sentiment analysis
filtered_df['sentiment'] = filtered_df['Text'].apply(get_sentiment)

# Save the result to a new Excel file
output_file_path = 'C:/Users/user/OneDrive/Documents/Advanced Willy/Final Data/Presentation/solaregypt/sorted_sentiment_analysis_filtered.xlsx'  # Replace with your desired output file path
filtered_df.to_excel(output_file_path, index=False)

print("Sentiment analysis complete. The results are saved to", output_file_path)

# Calculate a rolling average to smooth the data
filtered_df['rolling_sentiment'] = filtered_df['sentiment'].rolling(window=7).mean()  # 7-day rolling average

# Plotting the sentiment trend chart
plt.figure(figsize=(14, 7), dpi=300)
plt.plot(filtered_df['Date'], filtered_df['rolling_sentiment'], marker='o', linestyle='-', color='b', label='7-Day Rolling Average')
plt.fill_between(filtered_df['Date'], filtered_df['rolling_sentiment'], color='b', alpha=0.1)
plt.title('Sentiment Trend Over Time (Filtered Data)')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
plt.savefig('Sentiment.png')
plt.show()

df['Month'] = df['Date'].dt.to_period('M')
monthly_frequency = df['Month'].value_counts().sort_index()
print("Monthly frequency of texts:")
print(monthly_frequency)


df1 = pd.read_excel('C:/Users/user/OneDrive/Documents/Advanced Willy/NUFDI/NUFDI/Frequencies.xlsx')

plt.figure(figsize=(14, 10), dpi=300)
df1.plot(x='Month', y='Number', kind = 'line')
plt.xlabel('Month')
plt.ylabel('Number of Articles Publishe')
plt.title("Frequency of Article Publish by Month")
plt.xticks(rotation=45)
plt.savefig('Frequencies.png')