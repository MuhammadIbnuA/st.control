import streamlit as st
from google_play_scraper import Sort, reviews
from time import sleep
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import edit_distance
import nltk

# Define keywords as a global variable
keywords = ['Data', 'Rekaman', 'informasi', 'detail', 'fakta', 'angka', 'statistik',
            'Sistem', 'Struktur', 'kerangka', 'organisasi', 'pengaturan', 'metode',
            'Informasi', 'Intelijen', 'pengetahuan', 'berita', 'pembaruan', 'wawasan',
            'Keamanan', 'Keselamatan', 'perlindungan', 'keamanan', 'pertahanan', 'pengamanan',
            'Akses', 'Jalan masuk', 'jangkauan', 'pendekatan', 'koneksi', 'penerimaan',
            'Terjamin', 'Dijamin', 'diyakinkan', 'aman', 'pasti', 'dikonfirmasi',
            'Penggunaan', 'Penggunaan', 'pekerjaan', 'pemanfaatan', 'implementasi',
            'Error', 'Kesalahan', 'kegagalan', 'cacat', 'kekurangan', 'ketidakakuratan',
            'Terlindungi', 'Dijaga', 'dilindungi', 'dipelihara', 'dipertahankan', 'diamankan',
            'Diakses', 'Diakses', 'diperoleh', 'diambil', 'dikumpulkan',
            'Mudah', 'Sederhana', 'mudah dilakukan', 'langsung', 'tidak rumit', 'nyaman',
            'Konsumen', 'Pelanggan', 'pengguna', 'pembeli', 'pemborong',
            'Menggunakan', 'Memanfaatkan', 'menggunakan', 'menerapkan', 'menjalankan', 'mengoperasikan',
            'Aman', 'Aman', 'terjamin', 'terlindungi', 'terpercaya',
            'Hak', 'Hak', 'hak akses', 'wewenang', 'kuasa', 'prerogatif',
            'Batasan', 'Batasan', 'pembatasan', 'kendala', 'garis batas', 'pedoman',
            'Kecurangan', 'Penipuan', 'pengelabuan', 'tipu daya', 'ketidakjujuran', 'pelanggaran',
            'Pribadi', 'Personal', 'pribadi', 'rahasia', 'sensitif', 'intim',
            'Berbagai', 'Beragam', 'bervariasi', 'beberapa', 'berbagai macam', 'bermacam-macam',
            'Kejahatan', 'Kejahatan', 'pelanggaran', 'kesalahan', 'pelanggaran ringan', 'pelanggaran',
            'Keamanan Data', 'Keamanan data', 'perlindungan data', 'keamanan informasi',
            'Konsumen terlindungi', 'Konsumen terlindungi', 'konsumen aman',
            'Hak Akses', 'Hak akses', 'izin akses',
            'Data Identitas', 'Data identitas', 'informasi pribadi',
            'Data Transaksi', 'Data transaksi', 'data keuangan',
            'Pengendalian Akses', 'Kontrol akses', 'manajemen otorisasi',
            'Data Transaksi Konsumen', 'Data transaksi konsumen', 'data pembelian pelanggan',
            'Data Sistem Informasi', 'Data sistem informasi', 'data sistem IT',
            'Data Sensitif', 'Data sensitif', 'data rahasia', 'data pribadi',
            'Privasi', 'Privasi', 'kerahasiaan', 'pengasingan', 'kebijaksanaan', 'anonimitas',
            'Integritas', 'Integritas', 'akurasi', 'kelengkapan', 'konsistensi', 'keterandalan',
            'Keandalan', 'Keandalan', 'ketergantungan', 'kepercayaan', 'stabilitas', 'konsistensi',
            'Ketersediaan', 'Ketersediaan', 'aksesibilitas', 'kemampuan diambil kembali', 'kepraktisan',
            'Kepatuhan', 'Kepatuhan', 'kepenggunaan', 'kesesuaian', 'pematuhan', 'peraturan',
            'Penegakan', 'Penegakan', 'pelaksanaan', 'eksekusi', 'penuntutan',
            'Pengawasan', 'Pemantauan', 'pengawasan', 'pengintaian', 'supervisi', 'kontrol',
            'Manajemen', 'Manajemen', 'administrasi', 'tata kelola', 'kontrol', 'koordinasi',
            'Kebijakan', 'Kebijakan', 'prosedur', 'pedoman', 'aturan', 'peraturan',
            'Teknologi', 'Teknologi', 'alat', 'sistem', 'solusi', 'infrastruktur']

def scrape_reviews_batched(app_id, count=400, lang='id', country='id', sort=Sort.NEWEST, filter_score_with=""):
    all_reviews_content = []
    collected_review_ids = set()  # Set to store unique review IDs

    for _ in range(9):  # Scrape 9 batches (adjust as needed)
        result, continuation_token = reviews(app_id, lang=lang, country=country, sort=sort, count=count, filter_score_with=filter_score_with)
        
        # Append only review content to all_reviews_content
        for review in result:
            if review['reviewId'] not in collected_review_ids:
                all_reviews_content.append(review['content'])
                collected_review_ids.add(review['reviewId'])

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

def filter_reviews_by_keywords_cosine(reviews, keywords, threshold):
    # Initialize CountVectorizer
    vectorizer = CountVectorizer()
    # Fit vectorizer on keywords
    keyword_vectors = vectorizer.fit_transform(keywords)

    filtered_reviews = []
    similarity_scores = []
    for review in reviews:
        # Normalize review text
        review = normalize_text(review)
        # Compute cosine similarity between review and keywords
        similarity = cosine_similarity(vectorizer.transform([review]), keyword_vectors)[0]
        # If similarity is above the threshold, consider it a match
        if max(similarity) > threshold:
            filtered_reviews.append(review)
            similarity_scores.append(max(similarity))
    return filtered_reviews, similarity_scores

def filter_reviews_by_keywords_jaccard(reviews, keywords, threshold):
    filtered_reviews = []
    similarity_scores = []
    
    for review in reviews:
        # Normalize review text
        review = normalize_text(review)
        
        # Calculate Jaccard similarity
        review_words = set(review.split())
        keyword_words = set(keywords)
        jaccard_similarity = len(review_words.intersection(keyword_words)) / len(review_words.union(keyword_words))
        
        # If Jaccard similarity is above the threshold, consider it a match
        if jaccard_similarity > threshold:
            filtered_reviews.append(review)
            similarity_scores.append(jaccard_similarity)
    
    return filtered_reviews, similarity_scores

def filter_reviews_by_keywords_sorensen_dice(reviews, keywords, threshold):
    filtered_reviews = []
    similarity_scores = []
    
    for review in reviews:
        # Normalize review text
        review = normalize_text(review)
        
        # Calculate Sorensen-Dice similarity
        review_words = set(review.split())
        keyword_words = set(keywords)
        sorensen_dice_similarity = 2 * len(review_words.intersection(keyword_words)) / (len(review_words) + len(keyword_words))
        
        # If Sorensen-Dice similarity is above the threshold, consider it a match
        if sorensen_dice_similarity > threshold:
            filtered_reviews.append(review)
            similarity_scores.append(sorensen_dice_similarity)
    
    return filtered_reviews, similarity_scores

def filter_reviews_by_keywords_levensthein(reviews, keywords, threshold):
    filtered_reviews = []
    similarity_scores = []
    
    for review in reviews:
        # Normalize review text
        review = normalize_text(review)
        
        # Calculate Levenshtein distance
        min_distance = min(edit_distance(review, keyword) for keyword in keywords)
        
        # Normalize distance by maximum possible distance
        normalized_distance = 1 - min_distance / max(len(review), len(max(keywords, key=len)))
        
        # If normalized distance is above the threshold, consider it a match
        if normalized_distance > threshold:
            filtered_reviews.append(review)
            similarity_scores.append(normalized_distance)
    
    return filtered_reviews, similarity_scores

def main():
    st.title("App Reviews Keyword Filter")
    
    st.write("Our Dataset Keyword is Indonesian")
    
    # Input for the app ID
    app_id = st.text_input("Enter the Google Play Store app ID:")

    # Input for count parameter
    count = st.number_input("Enter the number of reviews to fetch per batch:", value=400, min_value=1)

    # Input for lang parameter
    lang = st.text_input("Enter the language code (e.g., 'id' for Indonesian):")

    # Input for country parameter
    country = st.text_input("Enter the country code (e.g., 'id' for Indonesia):")

    # Option to choose similarity measure
    similarity_measure = st.radio("Choose Similarity Measure:", ("Cosine Similarity", "Jaccard Similarity", "Sorensen-Dice Similarity", "Levenshtein Distance"), index=0)

    # Input for threshold
    threshold = st.number_input("Enter Threshold:", min_value=0.0, max_value=1.0, step=0.01, value=0.01)


    if app_id:
        reviews_content = scrape_reviews_batched(app_id, count, lang, country)

        if st.button("Filter Reviews"):
            if similarity_measure == "Cosine Similarity":
                # Filter reviews using cosine similarity
                reviews_with_keywords, similarity_scores = filter_reviews_by_keywords_cosine(reviews_content, keywords, threshold)
            elif similarity_measure == "Jaccard Similarity":
                # Filter reviews using Jaccard similarity
                reviews_with_keywords, similarity_scores = filter_reviews_by_keywords_jaccard(reviews_content, keywords, threshold)
            elif similarity_measure == "Sorensen-Dice Similarity":
                # Filter reviews using Sorensen-Dice similarity
                reviews_with_keywords, similarity_scores = filter_reviews_by_keywords_sorensen_dice(reviews_content, keywords, threshold)
            else:
                # Filter reviews using Levenshtein distance
                reviews_with_keywords, similarity_scores = filter_reviews_by_keywords_levensthein(reviews_content, keywords, threshold)

            # Create a DataFrame with review numbers, reviews, and similarity scores
            df_reviews_with_keywords = pd.DataFrame({"Review Number": range(1, len(reviews_with_keywords)+1),
                                                     "Review": reviews_with_keywords,
                                                     "Similarity Score": similarity_scores})

            # Display filtered reviews in a table
            st.write("Reviews containing keywords:")
            st.write(df_reviews_with_keywords)

    # Footer content
    footer = """
    <footer style="background-color:#f8f9fa; padding:10px; position: fixed; left: 0; bottom: 0; width: 100%; text-align: center;">
        <p>Created by Xilk & Egg. Find me on <a href="https://github.com/MuhammadIbnuA" target="_blank">GitHub</a>.</p>
    </footer>
    """

    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
