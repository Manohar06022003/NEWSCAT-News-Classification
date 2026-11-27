# NEWSCAT – Automated News Classification System

## Project Overview

NEWSCAT is a hybrid machine learning and deep learning-based news classification system that automatically categorizes news articles into five different categories: Business, Entertainment, Politics, Sport, and Tech.

The system uses Natural Language Processing (NLP) techniques and four classification models to predict the category of a given news article. A majority voting mechanism combines the predictions to generate the final classification result.

## Objectives

- Automate the classification of news articles.
- Apply NLP techniques for text preprocessing and feature extraction.
- Compare predictions from multiple machine learning and deep learning models.
- Combine model predictions using majority voting.

## Technologies Used

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn, XGBoost
- **Deep Learning:** TensorFlow, Keras
- **NLP:** TF-IDF, Tokenization
- **Data Processing:** Pandas, NumPy

## Classification Models

The system uses four models:

1. Multinomial Naive Bayes
2. Random Forest
3. XGBoost
4. Convolutional Neural Network (CNN)

### Majority Voting

Each model independently predicts a news category. The system compares the four predictions and selects the category receiving the highest number of votes.

## Project Structure

```text
NEWSCAT/
│
├── app.py
├── News_CAT_Updated_Code.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── bbc_news_text_complexity_summarization.csv
│
└── models/
    ├── best_model.keras
    ├── mnb_model.pkl
    ├── rf_model.pkl
    ├── xgb_model.pkl
    └── tokenizer.pkl
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NEWSCAT-News-Classification.git
```

### 2. Navigate to the project directory

```bash
cd NEWSCAT-News-Classification
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

## How to Use

1. Launch the Streamlit application.
2. Enter or paste a news article into the text input area.
3. Click the **Predict News Category** button.
4. View the predictions from all four models.
5. View the final category generated through majority voting.

## News Categories

- Business
- Entertainment
- Politics
- Sport
- Tech

## Author

**Manohar M**

GitHub: https://github.com/Manohar06022003