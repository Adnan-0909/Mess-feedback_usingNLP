import pandas as pd

df = pd.read_csv("mess_data.csv")
df.columns = [
    "Timestamp",
    "Meal",
    "Rating",
    "Feedback",
    "Taste",
    "Quantity",
    "Hygiene",
    "Service",
    "Issues",
    "Suggestions"
]
print(df.columns)
print(df.head())
print(df['Rating'].unique())
df['full_text'] = df['Feedback'].fillna('') + " " + df['Issues'].fillna('')
print(df['full_text'].head())
def get_sentiment(rating):
    if rating == 3:
        return 'neutral'
    elif rating <= 2:
        return 'negative'
    else:
        return 'positive'
df['Sentiment'] = df['Rating'].apply(get_sentiment)
print(df[['Rating', 'Sentiment']].head())
print(df['Sentiment'].value_counts())
