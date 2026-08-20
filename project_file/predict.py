import joblib
from utils.preprocessing import preprocess_text

model = joblib.load("models/logistic_model.pkl")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

text = input("Enter Message :")

text = preprocess_text(text)

text = vectorizer.transform([text])

prediction = model.predict(text)


if prediction[0] == 1 :
    print("Spam")

else:
    print("Ham")
