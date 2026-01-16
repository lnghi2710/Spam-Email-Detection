import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("spam_mail_dataset.csv")

# Spam vs Ham distribution (Bar chart)
label_counts = df["label"].value_counts()

plt.figure(figsize=(6, 4))
bars = plt.bar(label_counts.index, label_counts.values)

plt.title("Spam vs Ham Distribution", fontsize=13)
plt.xlabel("Email Type", fontsize=11)
plt.ylabel("Number of Emails", fontsize=11)

# Hiển thị số lượng trên mỗi cột
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


#Length distribution (Histogram)
df["text_length"] = df["text"].astype(str).apply(len)

plt.figure(figsize=(7, 4))
plt.hist(df["text_length"], bins=50)

plt.title("Email Text Length Distribution", fontsize=13)
plt.xlabel("Number of Characters", fontsize=11)
plt.ylabel("Number of Emails", fontsize=11)

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

