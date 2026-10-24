import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample Dataset
data = {
    'text': [
        "Urgent! Your bank account is suspended. Click here to verify now: http://fakebank.com",
        "Dear customer, please update your password immediately to avoid account termination.",
        "Congratulations! You won a $1000 gift card. Claim here: http://bit.ly/claim-prize",
        "Hi Team, please find attached the meeting notes for today's discussion.",
        "Hey, are we still meeting for lunch today at 1 PM?",
        "Your Amazon order #12345 has been shipped. Track your package on our official site.",
        "Verify your PayPal login details immediately by clicking this link.",
        "Reminder: Complete your project task submission before tomorrow."
    ],
    'label': ['Phishing', 'Phishing', 'Phishing', 'Safe', 'Safe', 'Safe', 'Phishing', 'Safe']
}

# 1. Create DataFrame
df = pd.DataFrame(data)

# 2. Extract Features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Split Dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Train Naive Bayes Classifier Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
print("=== Phishing Email Detection Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Test with Custom User Input
def predict_email(email_text):
    email_vectorized = vectorizer.transform([email_text])
    prediction = model.predict(email_vectorized)
    return prediction[0]

if __name__ == "__main__":
    print("\n--------------------------------------------------")
    user_email = input("Enter an email text to analyze: ")
    result = predict_email(user_email)
    print(f"\nResult: The email is classified as ---> [{result}]")
    print("--------------------------------------------------")