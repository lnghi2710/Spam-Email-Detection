import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# data_check
print("                     data_check loading")
df = pd.read_csv("spam_mail_dataset.csv")

df = df.drop(columns=['Unnamed: 0'])

df.head()
df.info()
df['label'].value_counts()

print(df)


# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)
print("                 clean_text successful! \n")



# convert text to number for learning model
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label_num']
print("                change text to number successful! \n")



# divide train_data and test_data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("                   divide dataset successful! \n")



# use logistic regression
model = LogisticRegression(
    max_iter=1000,
    class_weight={0:1, 1:1.5}
)

model.fit(X_train, y_train)


# evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

confusion_matrix(y_test, y_pred) 


# confusion_matrix chart 
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix", fontsize=13)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)

plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center", fontsize=11)

plt.colorbar()
plt.tight_layout()
plt.show()



# Prediction Distribution chart
result_df = pd.DataFrame({
    "True Label": y_test,
    "Predicted Label": y_pred
})

true_counts = result_df["True Label"].value_counts().sort_index()
pred_counts = result_df["Predicted Label"].value_counts().sort_index()

x = np.arange(len(true_counts))
width = 0.35

plt.figure(figsize=(6, 4))
plt.bar(x - width/2, true_counts, width, label="True")
plt.bar(x + width/2, pred_counts, width, label="Predicted")

plt.xticks(x, ["Ham", "Spam"])
plt.xlabel("Label", fontsize=11)
plt.ylabel("Number of Emails", fontsize=11)
plt.title("True vs Predicted Label Distribution", fontsize=13)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# save model
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf.pkl")