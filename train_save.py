"""Train the emotion ANN exactly as in AI_OEP.ipynb and save everything the GUI needs.

Run once:   python train_save.py
Creates:    artifacts/model.keras, artifacts/tfidf.pkl, artifacts/metrics.json
"""
import json
import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

for pkg in ("punkt_tab", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
OUT = "artifacts"
BATCH = 256
os.makedirs(OUT, exist_ok=True)

# ---------- NLP part (same as notebook) ----------
df = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")
df = df.drop_duplicates(subset="text")
raw_rows = len(df)

stop_words = set(stopwords.words("english")) - {"not", "no", "never", "nor"}
lemm = WordNetLemmatizer()


def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    words = [lemm.lemmatize(w) for w in word_tokenize(text) if w not in stop_words]
    return " ".join(words)


df["clean_text"] = df["text"].apply(preprocess)

x_train, x_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)


# ---------- ANN part (same as notebook) ----------
class SparseGen(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch=BATCH, shuffle=True):
        super().__init__()
        self.X, self.y = X, np.asarray(y)
        self.batch, self.shuffle = batch, shuffle
        self.idx = np.arange(X.shape[0])
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.batch))

    def __getitem__(self, i):
        b = self.idx[i * self.batch:(i + 1) * self.batch]
        return self.X[b].toarray().astype("float32"), self.y[b]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


train_gen = SparseGen(x_train, y_train)
test_gen = SparseGen(x_test, y_test, shuffle=False)

model = Sequential([
    Input(shape=(5000,)),
    Dense(512, activation="relu"),
    Dropout(0.4),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dense(6, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
history = model.fit(train_gen, validation_data=test_gen, epochs=15, callbacks=[early_stop])

y_pred = np.argmax(model.predict(test_gen), axis=1)
report = classification_report(y_test, y_pred, target_names=EMOTIONS, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# ---------- Save ----------
model.save(f"{OUT}/model.keras")
with open(f"{OUT}/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

metrics = {
    "emotions": EMOTIONS,
    "rows_after_dedupe": int(raw_rows),
    "train_size": int(x_train.shape[0]),
    "test_size": int(x_test.shape[0]),
    "class_counts": {EMOTIONS[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
    "report": report,
    "confusion_matrix": cm.tolist(),
    "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    "params": int(model.count_params()),
}
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nDone. Accuracy: {report['accuracy']:.4f}. Saved to ./{OUT}/")
