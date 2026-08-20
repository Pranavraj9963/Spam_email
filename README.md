📧 Smart Email Spam Detection using ML, DL and NLP

A Deep Learning project I built to automatically detect whether a message is Spam or Ham (not spam), using NLP for text processing and an LSTM model for classification.

About the Project

The idea was to build a system that can read a message and figure out on its own whether it's spam or a genuine message — using NLP techniques to clean the text and an LSTM neural network to actually learn and classify.

Dataset

I used the spam.csv dataset, which has messages labeled as:

ham → 0 (normal message)
spam → 1 (spam message)
What I Did
Cleaned the text (removed unwanted characters, lowercased, removed stopwords)
Applied lemmatization and tokenization
Converted text into numerical sequences
Built an LSTM model (Embedding → LSTM → Linear → Output)
Trained it and evaluated using accuracy, precision, recall, F1, and confusion matrix

Example:

Original: "Congratulations! You have WON a FREE prize."
After preprocessing: "congratulation win free prize"
Why LSTM?

LSTM works well for text because it can understand the relationship and order between words in a sentence — which matters a lot for figuring out context.

Results

Accuracy: ~95.5%

The model correctly picks up on patterns like "free," "prize," "click here," etc., while still being able to tell apart genuine conversational messages.

Examples:

Input	Prediction
"Congratulations! You have won a free prize. Click here to claim it."	Spam
"Hey, are we meeting tomorrow for the project discussion?"	Ham
Tech Stack

Python · Pandas · NumPy · NLTK · Scikit-learn · PyTorch · Matplotlib · Seaborn

What's Next
Try Transformer-based models like BERT
Add a Streamlit/Flask web interface
Deploy using FastAPI
Real-time email classification
Compare LSTM with GRU and Transformer models
