# Sentiment Analysis (IMDb Reviews)

## Project Goal
Build a machine learning model to classify movie reviews as positive or negative.

## Dataset
IMDb 50K movie reviews dataset.

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