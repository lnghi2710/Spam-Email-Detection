import joblib

# load model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

# new input
text = ["hello, good morning sir my name is"]

# xử lý giống lúc train
X_new = vectorizer.transform(text)

# predict
prediction = model.predict(X_new)

print("Spam" if prediction[0] == 1 else "Ham")
