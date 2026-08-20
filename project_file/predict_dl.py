import torch
import torch.nn as nn
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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


def encode(sentence):

    tokens = sentence.split()

    ids = [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokens
    ]

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
            128,
            128,
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

        hidden = torch.cat(
            (
                hidden[-2],
                hidden[-1]
            ),
            dim=1
        )

        hidden = self.dropout(hidden)

        return self.fc(hidden).squeeze(1)


model = SpamLSTM(
    len(vocab)
)

model.load_state_dict(
    torch.load(
        "models/spam_lstm_best.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

while True:

    text = input("\nEnter Message : ")

    if text.lower() == "exit":

        break

    text = preprocess(text)

    encoded = encode(text)

    x = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        output = model(x)

        probability = torch.sigmoid(output).item()

    print("\nSpam Probability :", probability)

    if probability >= 0.5:

        print("Prediction : SPAM")

    else:

        print("Prediction : HAM")