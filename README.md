# Fake News Predictor Using LSTM RNN

A deep learning project that classifies news headlines as real or fake using an LSTM-based recurrent neural network. The notebook walks through data loading, preprocessing, word embedding, model training, evaluation, and testing on unseen sample headlines.

## Overview

Fake news detection is a common Natural Language Processing problem where the goal is to identify whether a news item is reliable or misleading. This project uses headline text from real and fake news datasets, converts the text into numerical sequences, and trains an LSTM model to perform binary classification.

## Features

- Loads real and fake news datasets from CSV files
- Labels real news as `1` and fake news as `0`
- Cleans and preprocesses text using regular expressions, stopword removal, stemming, and lemmatization
- Converts text into one-hot encoded sequences
- Applies sequence padding for fixed-length model input
- Builds an LSTM neural network with TensorFlow/Keras
- Evaluates the model using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
- Tests the trained model on custom unseen headlines

## Dataset

The notebook uses the Fake News Detection dataset containing two CSV files:

- `True.csv` - real news articles
- `Fake.csv` - fake news articles

In the notebook, the files are loaded from a Kaggle input path:

```python
df_true = pd.read_csv('/kaggle/input/datasets/emineyetm/fake-news-detection-datasets/News _dataset/True.csv')
df_false = pd.read_csv('/kaggle/input/datasets/emineyetm/fake-news-detection-datasets/News _dataset/Fake.csv')
```

If you run the notebook locally, update these paths to match where you saved the dataset.

## Model Architecture

The model is built using TensorFlow/Keras:

```text
Input -> Embedding -> LSTM -> Dense -> Output
```

Main configuration:

- Vocabulary size: `50000`
- Sequence length: `243`
- Embedding dimensions: `40`
- LSTM units: `100`
- Output activation: `sigmoid`
- Loss function: `binary_crossentropy`
- Optimizer: `adam`

## Results

The saved notebook run achieved the following results on the test split:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.9462 |
| ROC-AUC | 0.9866 |
| Precision | 0.95 |
| Recall | 0.95 |
| F1-score | 0.95 |

Confusion matrix:

```text
[[7380  374]
 [ 423 6640]]
```

The notebook also includes visualizations for:

- Confusion matrix
- ROC curve
- Precision-recall curve
- Model performance metrics

## Project Structure

```text
fake-news-predictor-using-lstm-rnn/
├── fake-news-predictor-using-lstm-rnn.ipynb
└── README.md
```

## Requirements

Install the required Python libraries before running the notebook:

```bash
pip install numpy pandas tensorflow nltk scikit-learn matplotlib seaborn tqdm
```

You may also need to download NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-predictor-using-lstm-rnn.git
cd fake-news-predictor-using-lstm-rnn
```

2. Install dependencies:

```bash
pip install numpy pandas tensorflow nltk scikit-learn matplotlib seaborn tqdm
```

3. Download the dataset and place `True.csv` and `Fake.csv` in your preferred data folder.

4. Open the notebook:

```bash
jupyter notebook fake-news-predictor-using-lstm-rnn.ipynb
```

5. Update the dataset paths in the notebook if running outside Kaggle.

6. Run all cells.

## Prediction Logic

The model outputs a probability between `0` and `1`.

- `1` means real news
- `0` means fake news

In the notebook, predictions are thresholded like this:

```python
y_pred = np.where(y_pred_probs > 0.6, 1, 0)
```

## Notes

- The current notebook mainly uses the news headline/title field for prediction.
- Model performance may change depending on dataset split, preprocessing choices, threshold value, and training environment.
- For production use, the model should be retrained, validated on newer data, and tested for bias and robustness.

## Author

Created as a machine learning project for fake news detection using LSTM RNN.
