# Sentiment Analysis (IMDb Reviews)

## Project Goal
Build a machine learning model to classify movie reviews as positive or negative.

## Dataset
Dataset is not included in the repository due to size.
Download it from Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Place the CSV file inside the `data/` folder.

## Method
- Text preprocessing (cleaning, stopword removal)
- TF-IDF vectorization
- Logistic Regression classifier

## Results
Test accuracy: ~0.88 - 0.90

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Train model:
   python src/train.py

3. Evaluate model:
   python src/evaluate.py

4. Predict custom review:
   python src/predict.py
