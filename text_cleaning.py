import re
from sklearn.feature_extraction.text import TfidfVectorizer

from load_data import df
def clean_text(text):
    text = text.lower()                         # lowercase
    text = re.sub(r'[^a-z\s]', '', text)        # remove numbers/punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    return text

df['cleaned_text'] = df['full_text'].apply(clean_text)
df = df[df['cleaned_text'].str.strip() != ""] #remove empty respon
df = df[df['cleaned_text'].str.split().str.len() > 2] # remve responses with less than 3 words
print(df[['full_text', 'cleaned_text']].head())
print(df[['cleaned_text']].head())
print("Total rows:", len(df))



vectorizer = TfidfVectorizer(
    max_features=300,   # limit features (good for small dataset)
    ngram_range=(1,2)    # include single words + word pairs
)

X = vectorizer.fit_transform(df['cleaned_text'])
print("TF-IDF feature matrix shape:", X.shape)