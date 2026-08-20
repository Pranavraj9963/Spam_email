import streamlit as st
import torch
import torch.nn as nn
import joblib
from utils.preprocessing import preprocess_text

import re
import nltk
from utils.database import save_prediction
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

st.set_page_config(
    page_title="Smart Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

st.title("Smart email spam Detection")

st.write('spam Detection using Machine Learning + Deep Learning + NLP')


st.sidebar.title("Model")

model_choice = st.sidebar.selectbox("Choose Model",
                                    ['Random Forest',"BiLSTM"])

email = st.text_area("Enter Email Content",height = 250) 

predict = st.button('Predict')


if predict:

    if email.strip() == "":
        st.warning("Please Enter Email Content")

    else:

        text = preprocess_text(email)

        if model_choice == "Random Forest":

            model = joblib.load("models/random forest.pkl")

            tfidf = joblib.load("models/tfidf_vectorizer.pkl")

            vector = tfidf.transform([text])

            prediction = model.predict(vector)[0]

            probability = model.predict_proba(vector)[0][1]

            if prediction == 1:

                st.error("SPAM EMAIL")

            else:

                st.success("HAM EMAIL")

            st.write(f"Spam Probability : {probability*100:.2f}%")

            st.progress(float(probability))

            save_prediction(email,
                            "Random Forest",
                            "SPAM" if prediction == 1 else "HAM",
                            probability)
        
        elif model_choice == "BiLSTM":

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
                )
            
            vocab = joblib.load(
                "models/vocab.pkl"
                )

            MAX_LEN = 100 
            
            def preprocess(text):

                text = text.lower()

                text = re.sub(
                        r"[^a-zA-Z]",
                        " ",
                        text
                     )

                words = text.split()

                words = [
                    lemmatizer.lemmatize(word)
                    for word in words
                        if word not in stop_words
                        ]

                return " ".join(words)

            text = preprocess(email)

            def encode(sentence):

                tokens = sentence.split()

                ids = []

                for token in tokens:

                    ids.append(
                        vocab.get(
                        token,
                        vocab["<UNK>"]
                        )
                    )

                ids = ids[:MAX_LEN]

                while len(ids) < MAX_LEN:

                    ids.append(
                        vocab["<PAD>"]
                        )

                return ids
            


            class SpamLSTM(nn.Module):

                def __init__(self, vocab_size):

                    super().__init__()

                    self.embedding = nn.Embedding(
                        vocab_size,
                        128,
                        padding_idx=0
                        )

                    self.lstm = nn.LSTM(
                        input_size=128,
                        hidden_size=128,
                        num_layers=2,
                        batch_first=True,
                        bidirectional=True,
                        dropout=0.3
                        )

                    self.dropout = nn.Dropout(0.5)

                    self.fc = nn.Linear(
                            256,
                            1
                        )

                def forward(self, x):

                    x = self.embedding(x)

                    output, (hidden, cell) = self.lstm(x)

                    forward_hidden = hidden[-2]

                    backward_hidden = hidden[-1]

                    hidden = torch.cat(
                        (forward_hidden, backward_hidden),
                        dim=1
                        )

                    hidden = self.dropout(hidden)

                    return self.fc(hidden).squeeze(1)
                

            dl_model = SpamLSTM(len(vocab))

            dl_model.load_state_dict(
            torch.load(
                "models/spam_lstm_best.pth",
                map_location=device
                )
            )

            dl_model.to(device)

            dl_model.eval()

            encoded = encode(text)

            x = torch.tensor(
                [encoded],
                dtype=torch.long
                ).to(device)

            with torch.no_grad():

                output = dl_model(x)

                probability = torch.sigmoid(
                        output
                        ).item()

                prediction = 1 if probability >= 0.5 else 0

                if prediction == 1:

                    st.error("SPAM Email")

                else:

                    st.success("HAM Email")

                st.write(
                f"Spam Probability : {probability*100:.2f}%"
                    )

                st.progress(float(probability))

                save_prediction(
                            email,
                            "BiLSTM",
                            "SPAM" if prediction == 1 else "HAM",
                            probability
                            )
