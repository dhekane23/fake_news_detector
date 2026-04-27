# Fake News Detection System using NLP & Machine Learning

A machine learning system that classifies news articles as **Fake** or **Real** using Natural Language Processing techniques and Scikit-learn models — with both a CLI and a Flask web interface.

---

## Demo

![Fake News Detector Web UI](static/demo.png)

---

## Features

- Clean and preprocess raw news text (stopword removal, stemming)
- Convert text to numerical vectors using **TF-IDF**
- Train and compare 3 ML models: **Logistic Regression**, **Naive Bayes**, **Linear SVM**
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix
- Save and load trained models using **pickle**
- Predict via **CLI** (interactive or single-shot)
- **Flask web app** with a clean dark-themed UI

---

## Tech Stack

| Category        | Tools                              |
|-----------------|------------------------------------|
| Language        | Python 3.10+                       |
| ML / NLP        | Scikit-learn, Pandas, NumPy        |
| Visualization   | Matplotlib, Seaborn                |
| Web Framework   | Flask                              |
| Model Saving    | Pickle                             |
| Dataset         | Kaggle Fake and Real News Dataset  |

---

## Project Structure

```
fake_news_detector/
├── data/                   <- Place Fake.csv & True.csv here
├── models/                 <- Auto-generated .pkl files + confusion matrix PNGs
├── templates/
│   └── index.html          <- Flask web UI
├── static/                 <- Static assets
├── preprocess.py           <- Text cleaning pipeline
├── train.py                <- Train + evaluate + save model
├── predict.py              <- CLI inference
├── app.py                  <- Flask web app
├── demo.py                 <- Full pipeline demo (no dataset needed)
└── requirements.txt
```

---

## Dataset

Download the **Fake and Real News Dataset** from Kaggle:

> https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `Fake.csv` and `True.csv` inside the `data/` folder.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run demo (no dataset needed)
```bash
python demo.py
```

### 4. Train on real dataset
```bash
python train.py                  # Logistic Regression (default)
python train.py --model nb       # Naive Bayes
python train.py --model svm      # Linear SVM
```

### 5. Predict via CLI
```bash
# Interactive mode
python predict.py --model lr

# Single prediction
python predict.py --model lr --text "Government hiding alien contact since 1947"
```

### 6. Launch web app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## Model Performance

Trained on 44,898 news articles (Kaggle dataset):

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 98.81%   |
| Naive Bayes         | ~93%     |
| Linear SVM          | ~99%     |

Confusion matrix is auto-saved to `models/confusion_matrix_lr.png` after training.

---

## Example Predictions

| News Text                                              | Prediction  | Confidence |
|--------------------------------------------------------|-------------|------------|
| Central bank raises interest rates amid inflation      | REAL NEWS   | 91.3%      |
| Government putting mind-control chemicals in water     | FAKE NEWS   | 76.2%      |
| NASA launches new Mars rover for geological study      | REAL NEWS   | 88.7%      |
| Miracle pill cures cancer, Big Pharma hiding the truth | FAKE NEWS   | 74.9%      |

---

## Future Improvements

- Fine-tune a **BERT / DistilBERT** transformer model for higher accuracy
- Add **LIME / SHAP** explainability to highlight influential words
- Deploy to **AWS EC2 / Heroku** with a production WSGI server (Gunicorn)
- Add **multilingual** support using multilingual embeddings
- Integrate a **credibility score** based on news source domain

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Dhanashree**
- GitHub: [@dhanashree](https://github.com/dhanashree)
- LinkedIn: [dhanashree](https://linkedin.com/in/dhanashree)

---

## Acknowledgements

- Dataset by [Clement Bisaillon](https://www.kaggle.com/clmentbisaillon) on Kaggle
- Scikit-learn, Flask, and the open-source Python community
