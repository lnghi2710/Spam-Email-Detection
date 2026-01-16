import joblib

# load model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")

# mail text 
text = [
    "URGENT NOTICE!\n\nDear Customer,\n\nYour account has been temporarily suspended due to unusual activity detected on your system. To avoid permanent deactivation, you must verify your identity immediately.\n\nPlease click the link below and complete the verification within 24 hours:\nhttp://secure-verification-account-update.com\n\nFailure to act now will result in account termination and loss of access to all services.\n\nFor your security, please provide the required information including your full name, phone number, and verification code sent to your device.\n\nThank you for your prompt cooperation.\n\nSecurity Team"
]

# pre-processors
X_new = vectorizer.transform(text)

# predict
prediction = model.predict(X_new)

print("Spam" if prediction[0] == 1 else "Ham")
