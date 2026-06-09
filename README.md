[README.md](https://github.com/user-attachments/files/28767879/README.md)
# 🎬 Movie Review Sentiment Analysis using Deep Learning

Binary sentiment classification of movie reviews using Convolutional Neural Networks (CNNs) with pre-trained word embeddings.

Built as part of MANG3098 Analytics in Action II — University of Southampton.

---

## 📋 Overview

This project trains and evaluates four deep learning models to classify whether a movie review expresses a **positive** or **negative** sentiment. Two CNN architectures are compared across two pre-trained embedding types, producing four experimental configurations.

| Model | Accuracy | AUC-ROC |
|---|---|---|
| GloVe + Sequential CNN | 86.7% | 0.945 |
| **GloVe + Kim CNN** ✅ | **88.0%** | **0.951** |
| fastText + Sequential CNN | 86.6% | 0.942 |
| fastText + Kim CNN | 87.0% | 0.940 |

**Best model: GloVe + Kim CNN** — 88.0% accuracy on the held-out test set.

---

## 📁 Repository Structure

```
├── MANG3098_CW2_33649642.ipynb   # Main notebook — all code and outputs
├── README.md                      # This file
```

---

## 🧠 Models

### Architecture 1 — Sequential CNN
A stacked convolutional network adapted from course lab materials. Text is passed through three Conv1D layers that extract n-gram features at increasing levels of abstraction, followed by two dense hidden layers for classification.

### Architecture 2 — Kim CNN
A parallel multi-filter CNN based on Kim (2014). Four convolutional branches run simultaneously with filter widths of 2, 3, 4, and 5 tokens, capturing bigram through 5-gram patterns at once before concatenating into a single classification head.

---

## 🔤 Embeddings

| Embedding | Training Corpus | Dimensions | Vocabulary Coverage |
|---|---|---|---|
| GloVe | Wikipedia + Gigaword (6B tokens) | 300 | 99.2% |
| fastText | Common Crawl (600B tokens) | 300 | 100.0% |

Both embeddings were kept frozen during training to prevent overfitting.

---

## ⚙️ Setup and Usage

### Requirements

```
tensorflow>=2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
gdown
fasttext-wheel
```

### Running the Notebook

This notebook is designed to run on **Google Colab** with a GPU runtime.

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `MANG3098_CW2_33649642.ipynb` or open directly from GitHub
3. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
4. Run all cells top to bottom

> **Note:** GloVe vectors (~862 MB) and fastText Common Crawl model (~8 GB) are downloaded automatically when the relevant cells are run. Ensure you have sufficient Colab storage and a stable internet connection.

---

## 📊 Key Results

- All four models converged within 7–9 epochs due to strong initialisation from frozen pre-trained embeddings
- GloVe-based models consistently outperformed fastText equivalents despite fastText achieving higher vocabulary coverage, suggesting co-occurrence-based representations better capture sentiment-bearing vocabulary
- GloVe + Kim CNN's parallel architecture captured a wider range of phrase-level patterns, giving it the edge over the sequential model
- All models achieved AUC-ROC above 0.94, indicating strong discrimination across all classification thresholds

---

## 📚 References

- Kim, Y. (2014) *Convolutional Neural Networks for Sentence Classification*. EMNLP.
- Pennington, J., Socher, R. and Manning, C.D. (2014) *GloVe: Global Vectors for Word Representation*. EMNLP.
- Bojanowski, P. et al. (2017) *Enriching Word Vectors with Subword Information*. TACL.
- Maas, A.L. et al. (2011) *Learning Word Vectors for Sentiment Analysis*. ACL.

---

## 📝 Notes

- Random seed is set to student ID (33649642) throughout for full reproducibility
- Dataset sourced from the [Kaggle Word2Vec NLP tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)
- Report submitted separately as part of coursework assessment
