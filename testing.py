from text_cleaning import vectorizer, clean_text
from main import model

def predict_sentiment(feedback_text):
    """Predict sentiment for custom feedback"""
    cleaned = clean_text(feedback_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    return {
        'feedback': feedback_text,
        'cleaned': cleaned,
        'sentiment': prediction,
        'negative_prob': probability[0],
        'positive_prob': probability[1]
    }

# Test examples
test_inputs = [
    "The food tastes terrible and is always cold",
    "Great meal, very delicious!",
    "The hygiene could be better",
    "Amazing taste and good service",
    "Horrible experience, never coming back"
]

print("="*60)
print("SENTIMENT PREDICTION TEST")
print("="*60)

for feedback in test_inputs:
    result = predict_sentiment(feedback)
    print(f"\nFeedback: {result['feedback']}")
    print(f"Cleaned: {result['cleaned']}")
    print(f"Prediction: {result['sentiment'].upper()}")
    print(f"Confidence - Negative: {result['negative_prob']:.2%}, Positive: {result['positive_prob']:.2%}")