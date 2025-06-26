import logging
import os
import torch
from flask import Flask, render_template, request
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import time

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

global grade_color

# Headers for Amazon requests
headers = {
    'authority': 'www.amazon.in',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}

# Function to extract HTML data from multiple pages of reviews
def reviewsHtml(url, len_page=4):
    soups = []
    for page_no in range(1, len_page + 1):
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'filterByStar': 'critical',
            'pageNumber': page_no,
        }
        response = requests.get(url, headers=headers, params=params)
        soup = BeautifulSoup(response.text, 'lxml')
        soups.append(soup)
    return soups

# Function to parse review descriptions from HTML
def getReviews(html_data):
    data_dicts = []
    boxes = html_data.select('div[data-hook="review"]')
    for box in boxes:
        try:
            description = box.select_one('[data-hook="review-body"]').text.strip()
        except Exception:
            description = 'N/A'
        data_dict = {
            'review' : description
        }
        data_dicts.append({'review': description})
    return data_dicts

# Wrapper function to get Amazon reviews from multiple pages
def get_amazon_reviews(url, len_page=4):
    html_datas = reviewsHtml(url, len_page)
    reviews = []
    for html_data in html_datas:
        reviews += getReviews(html_data)
    return reviews

# Function to save reviews to a CSV file
def save_reviews_to_csv(reviews, filename):
    df = pd.DataFrame(reviews, columns=["review"])
    df.to_csv(filename, index=False)

# Function to load model and tokenizer
def load_model(model_path):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info(f"Model and tokenizer loaded from {model_path}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise ValueError(f"Error loading model from {model_path}: {e}")

# Function to process reviews and classify them
def process_reviews(file_path, model, tokenizer,model_name):
    df = pd.read_csv(file_path)
    if 'review' not in df.columns:
        raise ValueError("CSV file must contain 'review' column")

    predictions = []
    for idx, review in enumerate(df['review']):
        logging.info(f"Processing review {idx+1}/{len(df)}: {review[:50]}...")  # log part of review
        inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()  # 0 = real, 1 = fake
        predictions.append(prediction)
        logging.info(f"Prediction for review {idx+1}: {'Real' if prediction == 0 else 'Fake'}")
    
    df['prediction'] = predictions
    processed_filename = f'{model_name}_processed_reviews.csv'
    df.to_csv(processed_filename, index=False)
    # df.to_csv('processed_reviews.csv', index=False)

    real_count = (df['prediction'] == 0).sum()
    fake_count = (df['prediction'] == 1).sum()
    grade, grade_color = calculate_grade(real_count, fake_count)
    logging.info(f"Total Real Reviews: {real_count}, Total Fake Reviews: {fake_count}")
    return real_count, fake_count, grade

# Function to calculate grade based on review accuracy
def calculate_grade(real_count, fake_count):
    total_reviews = real_count + fake_count
    if total_reviews == 0:
        return "No reviews", "gray"
    
    real_percentage = (real_count / total_reviews) * 100
    if real_percentage >= 80:
        return 'A', 'green'
    elif real_percentage >= 60:
        return 'B', 'blue'
    elif real_percentage >= 40:
        return 'C', 'orange'
    elif real_percentage >= 20:
        return 'D', 'red'
    else:
        return 'F', 'darkred'

# Flask routes
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/BERT.html', methods=['GET','POST'])
def bert_view():
    grade_color = 'gray'  # Default color if no grade is assigned
    grade = "No reviews"  # Default grade if no reviews
    model_name = 'BERT'
    model_path = 'model_checkpoints/Bert'  # Path to the RoBERTa model
    model, tokenizer = load_model(model_path)
    logging.info(f"Processing reviews using model at {model_path}")
    
    if os.path.exists('BERT_processed_reviews.csv'):
        df = pd.read_csv('BERT_processed_reviews.csv')
        if 'prediction' in df.columns:
            real_count = (df['prediction'] == 0).sum()
            fake_count = (df['prediction'] == 1).sum()
            grade, grade_color = calculate_grade(real_count, fake_count)
            results = {
                "total_real_reviews": real_count,
                "total_fake_reviews": fake_count,
                "grade": grade,
                "grade_color": grade_color,
                "status_message": "File already processed."
            }
            logging.info(f"Results: {results}")
        else:
            url = request.form.get('url')
            reviews = get_amazon_reviews(url)
            save_reviews_to_csv(reviews, "amazon_reviews.csv")
            real_count, fake_count, grade = process_reviews("amazon_reviews.csv", model, tokenizer,model_name)
            grade_color = 'green'  # After processing, set the grade color based on the results
            results = {
                "total_real_reviews": real_count,
                "total_fake_reviews": fake_count,
                "grade": grade,
                "grade_color": grade_color,
                "status_message": "Processing completed."
            }
            logging.info(f"Processed reviews and computed grade: {results}")
    else:
        url = request.form.get('url')
        reviews = get_amazon_reviews(url)
        save_reviews_to_csv(reviews, "amazon_reviews.csv")
        real_count, fake_count, grade = process_reviews("amazon_reviews.csv", model, tokenizer, model_name)
        grade_color = 'green'  # After processing, set the grade color based on the results
        results = {
            "total_real_reviews": real_count,
            "total_fake_reviews": fake_count,
            "grade": grade,
            "grade_color": grade_color,
            "status_message": "Processing completed."
        }
        logging.info(f"Processed reviews and computed grade: {results}")

    return render_template('BERT.html', results=results, real_count=real_count, fake_count=fake_count)

@app.route('/XLnet.html', methods=['GET'])
def XLnet_view():
    grade_color = 'gray'  # Default color if no grade is assigned
    grade = "No reviews"  # Default grade if no reviews
    model_name = 'XLnet'

    model_path = 'model_checkpoints/XLnet'  # Path to the RoBERTa model
    model, tokenizer = load_model(model_path)
    logging.info(f"Processing reviews using model at {model_path}")
    
    if os.path.exists('XLnet_processed_reviews.csv'):
        df = pd.read_csv('XLnet_processed_reviews.csv')
        if 'prediction' in df.columns:
            real_count = (df['prediction'] == 0).sum()
            fake_count = (df['prediction'] == 1).sum()
            grade, grade_color = calculate_grade(real_count, fake_count)
            results = {
                "total_real_reviews": real_count,
                "total_fake_reviews": fake_count,
                "grade": grade,
                "grade_color": grade_color,
                "status_message": "File already processed."
            }
            logging.info(f"Results: {results}")
        else:
            real_count, fake_count, grade = process_reviews("amazon_reviews.csv", model, tokenizer, model_name)
            grade_color = 'green'  # After processing, set the grade color based on the results
            results = {
                "total_real_reviews": real_count,
                "total_fake_reviews": fake_count,
                "grade": grade,
                "grade_color": grade_color,
                "status_message": "Processing completed."
            }
            logging.info(f"Processed reviews and computed grade: {results}")
    else:
        real_count, fake_count, grade = process_reviews("amazon_reviews.csv", model, tokenizer,model_name)
        grade_color = 'green'  # After processing, set the grade color based on the results
        results = {
            "total_real_reviews": real_count,
            "total_fake_reviews": fake_count,
            "grade": grade,
            "grade_color": grade_color,
            "status_message": "Processing completed."
        }
        logging.info(f"Processed reviews and computed grade: {results}")

    return render_template('XLnet.html', results=results, real_count=real_count, fake_count=fake_count)

# @app.route('/detailed-view.html', methods=['GET', 'POST'])
# @app.route('/detailed-view.html', methods=['GET', 'POST'])
# def detailed_view():
    
#     return render_template('detailed-view.html')


if __name__ == '__main__':
    app.run(debug=True)
