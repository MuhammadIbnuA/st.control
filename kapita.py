import streamlit as st
from google_play_scraper import Sort, reviews
from time import sleep
import pandas as pd
import re

def scrape_reviews_batched(app_id, lang='id', country='id', sort=Sort.NEWEST, filter_score_with=""):
    all_reviews_content = []

    for _ in range(1):  # Scrape 9 batches (adjust as needed)
        result, continuation_token = reviews(app_id, lang=lang, country=country, sort=sort, count=200, filter_score_with=filter_score_with)
        
        # Append only review content to all_reviews_content
        all_reviews_content.extend(review['content'] for review in result)

        if not continuation_token:
            break  # No more pages to fetch, exit loop

        sleep(1)  # Delay for 1 second between batches

    return all_reviews_content

def normalize_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove symbols and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_reviews_by_keywords(reviews, keywords):
    filtered_reviews = []
    for review in reviews:
        for keyword in keywords:
            # Check if the review contains any of the keywords
            if re.search(r'\b{}\b'.format(re.escape(keyword)), review):
                filtered_reviews.append(review)
                break
    return filtered_reviews

def main():
    st.title("App Reviews Keyword Filter")
    
    # Input for the app ID
    app_id = st.text_input("Enter the Google Play Store app ID:")

    if app_id:
        reviews_content = scrape_reviews_batched(app_id)

        # Normalize each review content
        normalized_reviews_content = [normalize_text(review) for review in reviews_content]

        # keywords = st.text_area("Enter keywords (comma-separated):")
        keywords = ['data', 'Sistem', 'informasi', 'Keamanan', 'akses', 'terjamin', 'penggunaan', 'error', 'terlindungi', 'Diakses', 'Mudah', 'konsumen', 'menggunakan', 'aman', 'hak', 'batasan', 'kecurangan', 'pribadi', 'berbagai', 'kejahatan', 'dipahami', 'pengendalian', 'virus', 'digunakan', 'jelas']

        if st.button("Filter Reviews"):
            # Filter reviews by keywords
            reviews_with_keywords = filter_reviews_by_keywords(normalized_reviews_content, keywords)

            # Create a DataFrame with review numbers
            df_reviews_with_keywords = pd.DataFrame({"Review Number": range(1, len(reviews_with_keywords)+1),
                                                     "Review": (reviews_with_keywords)})

            # Display filtered reviews in a table
            st.write("Reviews containing keywords:")
            st.write(df_reviews_with_keywords)

if __name__ == "__main__":
    main()
