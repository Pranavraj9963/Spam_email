# import os
# import joblib 
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from utils.preprocessing import preprocess_text
# from sklearn.metrics import accuracy_score
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

# from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix,classification_report

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# print(device)


# df = pd.read_csv("dataset/spam.csv",encoding ="latin-1")


# df = df[['v1','v2']]

# df.columns = ['label','message']



# df['message'] = df['message'].apply(preprocess_text)

# df['label'] = df['label'].map({
#     'ham':0,
#     'spam':1
# })

# x_train,x_test,y_train,y_test = train_test_split(
#     df['message'],
#     df['label'],
#     test_size=0.2,
#     random_state=42,
#     stratify=df['label']
# )

# def tokenize(text):
#     return text.split()


# vocab ={
#     '<PAD>':0,
#     '<UNK>':1
# }

# for sentence in x_train:
#     for word in tokenize(sentence):
#         if word not in vocab :
#             vocab[word] = len(vocab)



    
# joblib.dump(vocab,"models/vocab.pkl")

# MAX_LEN = 100

# def encode(sentence):
#     tokens = tokenize(sentence)

#     ids = []
#     for token in tokens:
#         ids.append(vocab.get(token,vocab['<UNK>']))

#     ids = ids[:MAX_LEN]

#     while len(ids) < MAX_LEN:
#         ids.append(vocab["<PAD>"])

#     return ids
    



# class spamDataset(Dataset):
#     def __init__(self,texts,labels):
#         self.texts = texts
#         self.labels = labels


#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, index):
#         x = torch.tensor(encode(self.texts[index]),dtype = torch.long)

#         y = torch.tensor(self.labels[index],dtype = torch.float32)

#         return x, y


# train_dataset = spamDataset(x_train.tolist(),y_train.tolist())

# test_dataset = spamDataset(x_test.tolist(),y_test.to_list())


# train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

# test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)



# class spamLSTM(nn.Module):
#     def __init__(self,vocab_size):
#         super().__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings = vocab_size,
#             embedding_dim = 128,
#             padding_idx =  0
#         )

#         self.lstm = nn.LSTM(
#             input_size = 128,
#             hidden_size = 64,
#             batch_first = True
#         )

#         self.fc = nn.Linear(64,1)

#     def forward(self,x):
#         x = self.embedding(x)

#         output,(hidden ,cell) = self.lstm(x)
#         hidden = hidden.squeeze(0)
#         output = self.fc(hidden)

#         return output.squeeze(1)
    


# model = spamLSTM(
#     vocab_size=len(vocab)
# )

# model.to(device)

# criterion = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

# Epochs = 10

# for epoch in range(Epochs):
#     model.train()

#     total_loss = 0

#     for x_batch,y_batch in train_loader:
#         x_batch = x_batch.to(device)
#         y_batch = y_batch.to(device)

#         optimizer.zero_grad()

#         outputs = model(x_batch)
         
#         loss = criterion(outputs , y_batch)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     avg_loss = total_loss / len(train_loader)

#     print(f"Epoch {epoch+1}/{Epochs} | Average Loss : {avg_loss:.4f}")



# model.eval()

# all_predictions = []
# all_labels = []

# with torch.no_grad():
#         for x_batch,y_batch in test_loader:
#             x_batch = x_batch.to(device)

#             outputs = model(x_batch)

#             predictions = torch.sigmoid(outputs)

#             predictions = (predictions >= 0.5).int()

#             all_predictions.extend(predictions.cpu().numpy())

#             all_labels.extend(y_batch.numpy())


# accuracy = accuracy_score(all_labels,all_predictions)

# precision = precision_score(all_labels,all_predictions)

# recall = recall_score(all_labels,all_predictions)

# f1 = f1_score(all_labels,all_predictions)


# print("="*60)

# print(f"Accuracy  : {accuracy:.4f}")

# print(f"Precision : {precision:.4f}")

# print(f"Recall    : {recall:.4f}")

# print(f"F1 Score  : {f1:.4f}")

# print("\n Classification Report\n")

# print(classification_report(all_labels,all_predictions))


# cm = confusion_matrix(all_labels,all_predictions)

# print("\n Confusion Matrix\n")

# print(cm)



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay
import os
import re
import joblib
import pandas as pd
import nltk

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ============================================
# Device
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", device)

# ============================================
# Download NLTK
# ============================================

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ============================================
# Text Preprocessing
# ============================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv(
    "dataset/spam.csv",
    encoding="latin-1"
)

df = df[["v1", "v2"]]

df.columns = [
    "label",
    "message"
]

df["label"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1
    }
)

df["message"] = df["message"].apply(
    preprocess_text
)

print(df.head())

print()

print(df["label"].value_counts())

# ============================================
# Train Test Split
# ============================================

x_train, x_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ============================================
# Vocabulary
# ============================================

def tokenize(sentence):
    return sentence.split()

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for sentence in x_train:

    for word in tokenize(sentence):

        if word not in vocab:

            vocab[word] = len(vocab)

print()

print("Vocabulary Size :", len(vocab))

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    vocab,
    "models/vocab.pkl"
)

# ============================================
# Encoding
# ============================================

MAX_LEN = 100

def encode(sentence):

    tokens = tokenize(sentence)

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

print()

print(x_train.iloc[0])

print()

print(encode(x_train.iloc[0])[:20])

# ============================================
# Dataset
# ============================================

class SpamDataset(Dataset):

    def __init__(
        self,
        texts,
        labels
    ):

        self.texts = texts

        self.labels = labels

    def __len__(self):

        return len(
            self.texts
        )

    def __getitem__(self, index):

        x = torch.tensor(
            encode(
                self.texts[index]
            ),
            dtype=torch.long
        )

        y = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return x, y

train_dataset = SpamDataset(
    x_train.tolist(),
    y_train.tolist()
)

test_dataset = SpamDataset(
    x_test.tolist(),
    y_test.tolist()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print()

print("Train Samples :", len(train_dataset))

print("Test Samples :", len(test_dataset))


# ============================================
# MODEL
# ============================================

class SpamLSTM(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=128,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            128 * 2,
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

        out = self.fc(hidden)

        return out.squeeze(1)


# ============================================
# MODEL OBJECT
# ============================================

model = SpamLSTM(
    vocab_size=len(vocab)
).to(device)

print(model)

# ============================================
# CLASS WEIGHT
# ============================================

spam_count = (df["label"] == 1).sum()

ham_count = (df["label"] == 0).sum()

print("Ham :", ham_count)
print("Spam:", spam_count)

pos_weight = torch.tensor(
    [ham_count / spam_count],
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

# ============================================
# OPTIMIZER
# ============================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

# ============================================
# LR SCHEDULER
# ============================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ============================================
# TRAINING
# ============================================

EPOCHS = 15

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for x_batch, y_batch in train_loader:

        x_batch = x_batch.to(device)

        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(x_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    scheduler.step(avg_loss)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss : {avg_loss:.4f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "models/spam_lstm_best.pth"
        )

print()

print("Training Finished")

print("Best Loss :", best_loss)


# ============================================
# LOAD BEST MODEL
# ============================================

print("\nLoading Best Model...\n")

model.load_state_dict(
    torch.load(
        "models/spam_lstm_best.pth",
        map_location=device
    )
)

model.eval()

# ============================================
# EVALUATION
# ============================================

all_predictions = []
all_probabilities = []
all_labels = []

with torch.no_grad():

    for x_batch, y_batch in test_loader:

        x_batch = x_batch.to(device)

        outputs = model(x_batch)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).int()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_labels.extend(
            y_batch.numpy()
        )

# ============================================
# METRICS
# ============================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("=" * 60)

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        zero_division=0
    )
)

# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix\n")

print(cm)

# ============================================
# SAMPLE PREDICTIONS
# ============================================

print("\nSample Predictions\n")

for i in range(10):

    label = "SPAM" if all_predictions[i] == 1 else "HAM"

    print(
        f"{i+1}. Probability = "
        f"{all_probabilities[i]:.4f}"
        f" ---> {label}"
    )

print()


auc = roc_auc_score(all_labels,all_probabilities)
print(f"AUC Score: {auc:4f}")


fpr,tpr, thresholds = roc_curve(all_labels,all_probabilities)

plt.figure(figsize = (6,6))

plt.plot(fpr,tpr,label =f"AUC = {auc:4f}")
plt.plot([0,1],
         [0,1],
         "--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()


disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()