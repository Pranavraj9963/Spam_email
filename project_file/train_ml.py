import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from utils.preprocessing import preprocess_text

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

df = pd.read_csv("dataset/spam.csv",encoding ="latin-1")

df = df[['v1','v2']]
df.columns = ['label','message']

df['message'] =df['message'].apply(preprocess_text)
df['label'] = df['label'].map({
    "ham":0,
    "spam":1
})



x = df['message']
y = df['label']


vectorizer = TfidfVectorizer(max_features=5000)

x = vectorizer.fit_transform(x)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state= 42)

models ={ "Logistic Regression":LogisticRegression(max_iter=1000),
         "Naive Bayes": MultinomialNB(),
         "Random forest":RandomForestClassifier(n_estimators=100,random_state=42)
         }


os.makedirs("models",exist_ok = True)

for name,model in models.items():
    print("="* 60)
    print(f"Training{name}")

    model.fit(x_train,y_train)

    prediction = model.predict(x_test)

    accuracy = accuracy_score(y_test,prediction)
    precision = precision_score(y_test,prediction)
    recall = recall_score(y_test,prediction)
    f1 = f1_score(y_test,prediction)


    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test,prediction))

    joblib.dump(
        model,
        f"models/{name.replace(' ',' ').lower()}.pkl"
    )

    joblib.dump(vectorizer,"models/tfidf_vectorizer.pkl")