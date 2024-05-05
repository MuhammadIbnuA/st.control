import streamlit as st
from google_play_scraper import Sort, reviews
from time import sleep
import pandas as pd
import re

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
    
    st.write("Our Dataset Keyword is Indonesian")
    
    # Input for the app ID
    app_id = st.text_input("Enter the Google Play Store app ID:")

    # Input for count parameter
    count = st.number_input("Enter the number of reviews to fetch per batch:", value=400, min_value=1)

    # Input for lang parameter
    lang = st.text_input("Enter the language code (e.g., 'id' for Indonesian):")

    # Input for country parameter
    country = st.text_input("Enter the country code (e.g., 'id' for Indonesia):")

    if app_id:
        reviews_content = scrape_reviews_batched(app_id, count, lang, country)

        # Normalize each review content
        normalized_reviews_content = [normalize_text(review) for review in reviews_content]

        # keywords = st.text_area("Enter keywords (comma-separated):")
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
        if st.button("Filter Reviews"):
            # Filter reviews by keywords
            reviews_with_keywords = filter_reviews_by_keywords(normalized_reviews_content, keywords)

            # Create a DataFrame with review numbers
            df_reviews_with_keywords = pd.DataFrame({"Review Number": range(1, len(reviews_with_keywords)+1),
                                                     "Review": (reviews_with_keywords)})

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
