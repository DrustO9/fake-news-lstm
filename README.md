# fake-news-lstm

Binary fake-vs-real news classifier built on a single **LSTM** layer over word-embedding
representations of headlines. Trained on the Kaggle
[`emineyetm/fake-news-detection-datasets`](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
(~45k articles, balanced True/Fake).

## What it does

1. Loads `True.csv` and `Fake.csv`, labels them `1 / 0`, concatenates and shuffles.
2. Cleans titles: regex strip non-alpha, lowercase, lemmatise, drop English stopwords.
3. One-hot encodes each cleaned title against a **50k-word vocabulary**, pads to length **243**.
4. Trains an LSTM:
   ```
   Input(243) → Embedding(50000, 40) → LSTM(100) → Dense(1, sigmoid)
   ```
   ~**2.05M parameters**, `binary_crossentropy` loss, Adam optimiser, 20 epochs, batch 64,
   67/33 train/test split.
5. Evaluates with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, PR curve, and
   tests on **unseen / adversarial headlines** at the end.

## Why it's interesting

The notebook walks through *why* word embeddings + LSTM beat TF-IDF on this task — a side-by-side
table explains how dense vectors capture context and word order that bag-of-words models can't.
It's a good worked example of "when does the sequence matter?" in NLP.

## Tech stack

- **TensorFlow 2 / Keras** — `Embedding`, `LSTM`, `Dense`, `Sequential`
- **NLTK** — `stopwords`, `WordNetLemmatizer`, `PorterStemmer`
- **scikit-learn** — `train_test_split`, all metrics, `classification_report`
- **pandas, NumPy** — data handling
- **matplotlib, seaborn** — ROC, PR, confusion matrix
- Built on Kaggle's `kaggle/python` Docker image

## How to run

```bash
pip install tensorflow nltk scikit-learn pandas numpy matplotlib seaborn jupyter
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

Open `fake-news-predictor-using-lstm-rnn.ipynb`, repoint the two `pd.read_csv(...)` calls
at your local copies of `True.csv` and `Fake.csv`, then `Run All`.

## Key results

| Metric | Value |
|---|---|
| Test accuracy | **~95 %** |
| F1 | **0.95** |
| ROC-AUC | ≈ 0.99 |
| Params | 2.05M |

> 📷 *Plots in the notebook: confusion matrix, ROC curve, precision-recall curve, 4-panel summary.*

## License

MIT (add a `LICENSE` file if you want this enforced).
