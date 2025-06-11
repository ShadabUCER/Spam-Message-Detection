import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = [
    ["spam", "Congratulations! You've won a free ticket to Bahamas. Call now!"],
    ["ham", "Are we meeting tomorrow at 10 am?"],
    ["spam", "Claim your $1000 gift card now! Limited time offer."],
    ["ham", "Can you please send me the notes from today's class?"],
    ["spam", "Urgent! Your account has been suspended. Verify now."],
    ["ham", "Hey, I’ll call you later in the evening."],
    ["spam", "You are selected for a free laptop. Click here to claim."],
    ["ham", "Let's go for dinner tonight if you're free."],
    ["spam", "Win a brand new iPhone just by entering your number."],
    ["ham", "Don’t forget to bring your ID card tomorrow."],
    ["spam", "Lowest prices on medicines. Buy now and save big!"],
    ["ham", "I'll be home by 6 pm today."],
    ["spam", "You have been chosen for a lucky draw prize!"],
    ["ham", "Thanks for the help earlier today."],
    ["ham", "Reminder: Doctor's appointment at 4:00 PM tomorrow."]
]

# Save/load model
model_file = "spam_model.pkl"
vector_file = "vectorizer.pkl"

if os.path.exists(model_file) and os.path.exists(vector_file):
    model = joblib.load(model_file)
    vectorizer = joblib.load(vector_file)
else:
    df = pd.DataFrame(data, columns=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    X = df["message"]
    y = df["label"]

    vectorizer = CountVectorizer()
    X_vector = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vector, y)

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vector_file)

# Function to predict
def check_message():
    msg = entry.get("1.0", tk.END).strip()
    if msg == "":
        messagebox.showwarning("Empty Input", "Please enter a message.")
        return
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    messagebox.showinfo("Result", f"The message is: {result}")

# GUI
root = tk.Tk()
root.title("Spam Message Detector")
root.geometry("500x350")
root.configure(bg="#f0f0f0")

title = tk.Label(root, text="🧠 Spam Message Detector", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
title.pack(pady=10)

label = tk.Label(root, text="Enter your message:", font=("Arial", 13), bg="#f0f0f0")
label.pack()

entry = tk.Text(root, height=5, width=55, font=("Courier", 10))
entry.pack(pady=5)

button = tk.Button(root, text="Check", command=check_message, font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
button.pack(pady=20)

root.mainloop()
