import streamlit as st
from google_play_scraper import Sort, reviews
from time import sleep
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import edit_distance
from googletrans import Translator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Define keyword categories
performance_keywords = ['performance', 'speed', 'efficiency', 'Penggunaan', 'Ketersediaan', 'Menggunakan', 'Mengevaluasi', 'Memperhatikan', 'Mudah', 'Cepat', 'Meningkatkan', 'Mudah diakses', 'Diakses', 'Baik', 'Mudah dipahami', 'Meningkatkan kepuasan', 'Sesuai', 'Cepat mudah', 'Tepat waktu', 'Waktu hasil', 'Dipahami', 'Digunakan', 'Dipahami mudah', 'Pengguna', 'Mudah digunakan', 'Digunakan mudah', 'Dijalankan', 'Secara', 'Stabil', 'Merespon', 'Tampilan', 'Memuaskan', 'Baik sesuai', 'Meningkatkan produktivitas', 'Tepat', 'Hasil', 'Kepuasan', 'Efisien']
information_keywords = ['information', 'data', 'report', 'Diandalkan memberikan', 'Memberikan notifikasi', 'Cepat diperoleh', 'Memerlukan proses', 'Sesuai kebutuhan', 'Memiliki ketepatan', 'Menyimpan data', 'Proses input', 'Dipercaya akurat', 'Keputusan cepat', 'Menunjukkan cepat', 'Diperoleh informasi', 'Duplikasi bermanfaat', 'Sesuai bermanfaat', 'Kebutuhan membantu', 'Dibaca memerlukan', 'Diketahui memerlukan', 'Memerlukan input', 'Diperoleh informasi', 'Merilis informasi', 'Tersimpan memperoleh', 'Diverifikasi tersimpan', 'Tersimpan efektif', 'Akurat relevan', 'Date mencegah', 'Mencegah akurasi', 'Feedback menghasilkan', 'Menghasilkan keluaran', 'Keluaran kehandalan', 'Kehandalan memasukkan']
economy_keywords = [
    "ekonomi",
    "biaya",
    "anggaran",
    "ROI",
    "laba atas investasi",
    "efisiensi",
    "produktivitas",
    "keuntungan",
    "nilai tambah",
    "risiko",
    "keuangan",
    "pasar",
    "kompetisi",
    "permintaan",
    "penawaran",
    "harga",
    "inflasi",
    "pertumbuhan ekonomi",
    "kesejahteraan",
    "ketidaksetaraan",
    "kebijakan ekonomi",
    "sistem ekonomi",
    "model ekonomi",
    "analisis ekonomi",
    "data ekonomi",
    "prediksi ekonomi",
    "simulasi ekonomi",
    "manajemen ekonomi",
    "kewirausahaan",
    "bisnis",
    "industri",
    "perusahaan",
    "startup",
    "investasi",
    "modal",
    "tenaga kerja",
    "teknologi",
    "inovasi",
    "pasar tenaga kerja",
    "globalisasi",
    "perdagangan internasional",
    "pembangunan ekonomi",
    "keberlanjutan",
    "lingkungan",
    "etika bisnis",
    "tanggung jawab sosial",
    "tata kelola perusahaan",
    "regulasi",
    "kebijakan publik",
]

control_keywords = ['Data', 'Rekaman', 'informasi', 'detail', 'fakta', 'angka', 'statistik',
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
efficiency_keywords = [
    'efisiensi', 'produktivitas', 'optimasi',
    'kebutuhan', 'mengurangi', 'menggunakan', 'dibutuhkan', 'pembelajaran',
    'efisiensi energi', 'efisiensi waktu', 'efisiensi biaya', 'efisiensi proses',
    'efisiensi sumber daya', 'efisiensi operasi', 'efisiensi produksi',
    'efisiensi kerja', 'efisiensi manajemen', 'efisiensi logistik', 'efisiensi operasional',
    'potongan', 'bagian', 'komponen', 'modul', 'unit', 'elemen', 'segmen', 'blok',
    'fraksi', 'komponen', 'struktur', 'efektif', 'hemat', 'minimalisasi',
    'maksimalisasi', 'memanfaatkan', 'memperoleh', 'meningkatkan', 'mencapai'
]
service_keywords = ['service', 'support', 'help', "melayani", "menyediakan", "memberikan", "bertugas", "kepuasan", "kepuasan pelanggan", "kebahagiaan", "menanggapi", "merespons", "menindaklanjuti", "bertindak balas", "membatalkan", "pembatalan", "dibatalkan", "terjamin", "aman", "pasti", "dijamin", "keamanan", "security", "keselamatan", "keamanan", "data", "informasi", "fakta", "efisiensi", "efisien", "hemat", "optimal", "time", "waktu", "saat", "mudah", "sederhana", "simpel", "tidak sulit", "dengan mudah", "mudah digunakan", "sederhana", "gampang", "dengan cepat", "cepat", "kilat", "segera", "user", "pengguna", "konsumen", "memperlihatkan", "menunjukkan", "menampilkan", "ada", "tersedia", "ready", "murah", "hemat", "terjangkau", "ekonomis", "cocok", "tepat", "sesuai", "kualitas", "mutu", "keunggulan", "standar", "antarmuka", "interface", "ui (user interface)", "tampilan", "pengalaman", "pengalaman hidup", "pengalaman belajar", "aplikasi", "program", "software", "sistem", "fungsi", "kegunaan", "manfaat", "tujuan", "proses", "tahap", "langkah", "urutan","panduan", "manual", "buku petunjuk", "dipakai", "dimanfaatkan", "digunakan", "diakses", "diambil", "dijangkau", "dibahas", "diperbincangkan", "didiskusikan", "dibicarakan", "skenario", "scenario", "situasi", "rencana", "evaluasi", "penilaian", "assessment", "output", "hasil", "outcome", "memperjelas", "menyoroti", "menekankan", "berkaitan", "terhubung", "terkait", "pengetahuan", "knowledge", "pengetahuan", "ilmu", "pembelajaran", "pengajaran", "didikan", "metode", "cara", "metode", "teknik", "menilai", "menilai", "mengevaluasi", "menghargai", "topik", "subjek", "tema", "skill", "kemampuan", "keahlian", "job", "pekerjaan", "tugas", "memenuhi", "menyesuaikan", "memadai", "gift", "hadiah", "pemberian", "appearance", "penampilan", "tampilan", "situs web", "website", "situs internet", "portal"]

def scrape_reviews_batched(app_id, count=400, lang='id', country='id', sort=Sort.NEWEST, filter_score_with=None):
    all_reviews_content = []
    collected_review_ids = set()

    for _ in range(40):
        result, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort,
            count=count,
            filter_score_with=filter_score_with
        )

        for review in result:
            if review['reviewId'] not in collected_review_ids:
                all_reviews_content.append(review['content'])
                collected_review_ids.add(review['reviewId'])

        if not continuation_token:
            break

        sleep(1)

    return all_reviews_content


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def filter_reviews_by_keywords_cosine(reviews, keywords, threshold):
    vectorizer = CountVectorizer()
    keyword_vectors = vectorizer.fit_transform(keywords)

    filtered_reviews = []
    similarity_scores = []
    for review in reviews:
        review = normalize_text(review)
        similarity = cosine_similarity(vectorizer.transform([review]), keyword_vectors)[0]
        if max(similarity) > threshold:
            filtered_reviews.append(review)
            similarity_scores.append(max(similarity))
    return filtered_reviews, similarity_scores


def filter_reviews_by_keywords_jaccard(reviews, keywords, threshold):
    filtered_reviews = []
    similarity_scores = []

    for review in reviews:
        review = normalize_text(review)

        review_words = set(review.split())
        keyword_words = set(keywords)
        jaccard_similarity = len(review_words.intersection(keyword_words)) / len(review_words.union(keyword_words))

        if jaccard_similarity > threshold:
            filtered_reviews.append(review)
            similarity_scores.append(jaccard_similarity)

    return filtered_reviews, similarity_scores


def filter_reviews_by_keywords_sorensen_dice(reviews, keywords, threshold):
    filtered_reviews = []
    similarity_scores = []

    for review in reviews:
        review = normalize_text(review)

        review_words = set(review.split())
        keyword_words = set(keywords)
        dice_coefficient = 2 * len(review_words.intersection(keyword_words)) / (len(review_words) + len(keyword_words))

        if dice_coefficient > threshold:
            filtered_reviews.append(review)
            similarity_scores.append(dice_coefficient)

    return filtered_reviews, similarity_scores


def filter_reviews_by_keywords_levensthein(reviews, keywords, threshold):
    filtered_reviews = []
    similarity_scores = []

    for review in reviews:
        review = normalize_text(review)

        min_distance = min(edit_distance(review, keyword) for keyword in keywords)
        max_length = max(len(review), max(len(keyword) for keyword in keywords))
        levenshtein_similarity = 1 - (min_distance / max_length)

        if levenshtein_similarity > threshold:
            filtered_reviews.append(review)
            similarity_scores.append(levenshtein_similarity)

    return filtered_reviews, similarity_scores


def translate_reviews(reviews):
    translator = Translator()
    translated_reviews = [translator.translate(review, src='auto', dest='en').text for review in reviews]
    return translated_reviews


def analyze_sentiment_vader(reviews):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(review)["compound"] for review in reviews]
    return sentiment_scores


def classify_sentiment_vader(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def scale_sentiment_to_five_levels(sentiment):
    if sentiment <= -0.6:
        return "Kurang Puas"
    elif sentiment <= -0.2:
        return "Tidak Puas"
    elif sentiment <= 0.2:
        return "Cukup Puas"
    elif sentiment <= 0.6:
        return "Puas"
    else:
        return "Sangat Puas"


def analyze_and_visualize_category(reviews, category_name, keywords, threshold, similarity_measure):
    st.write(f"### Analysis for {category_name} Keywords")

    if similarity_measure == "Cosine":
        filtered_reviews, similarity_scores = filter_reviews_by_keywords_cosine(reviews, keywords, threshold)
    elif similarity_measure == "Jaccard":
        filtered_reviews, similarity_scores = filter_reviews_by_keywords_jaccard(reviews, keywords, threshold)
    elif similarity_measure == "Sorensen-Dice":
        filtered_reviews, similarity_scores = filter_reviews_by_keywords_sorensen_dice(reviews, keywords, threshold)
    elif similarity_measure == "Levenshtein":
        filtered_reviews, similarity_scores = filter_reviews_by_keywords_levensthein(reviews, keywords, threshold)

    translated_reviews = translate_reviews(filtered_reviews)
    sentiment_scores = analyze_sentiment_vader(translated_reviews)
    sentiment_labels = [classify_sentiment_vader(score) for score in sentiment_scores]

    results_df = pd.DataFrame({
        'Review': filtered_reviews,
        'Translated Review': translated_reviews,
        'Similarity Score': similarity_scores,
        'Sentiment Score': sentiment_scores,
        'Sentiment': sentiment_labels
    })

    st.write(f"Total reviews after filtering for {category_name}: {len(filtered_reviews)}")
    st.write(results_df)

    st.write(f"Sentiment distribution for {category_name}:")
    st.bar_chart(results_df['Sentiment'].value_counts())

    st.write(f"Sentiment scores distribution for {category_name}:")
    fig, ax = plt.subplots()
    ax.hist(sentiment_scores, bins=20, edgecolor='k')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write(f"Sentiment levels distribution for {category_name} (scaled to five levels):")
    results_df['Five-level Sentiment'] = results_df['Sentiment Score'].apply(scale_sentiment_to_five_levels)
    st.bar_chart(results_df['Five-level Sentiment'].value_counts())


def main():
    st.title("App Reviews Keyword Filter")

    app_id = st.text_input("Enter the Google Play Store app ID:")

    count = st.number_input("Enter the number of reviews to fetch per batch:", value=400, min_value=1)

    filter_score_input = st.text_input("Enter the score filter (1, 2, 3, 4, 5). Leave empty for no filter:", value='')
    filter_score_with = [int(score) for score in filter_score_input.split(',')] if filter_score_input else None

    threshold = st.slider("Select the keyword matching threshold:", min_value=0.0, max_value=1.0, value=0.4)

    similarity_measure = st.selectbox(
        "Select the similarity measure:",
        ["Cosine", "Jaccard", "Sorensen-Dice", "Levenshtein"]
    )

    if st.button("Scrape and Analyze Reviews"):
        with st.spinner("Scraping reviews..."):
            reviews = scrape_reviews_batched(app_id, count=count, lang='id', country='id',
                                             filter_score_with=filter_score_with)

        st.write(f"Total reviews fetched: {len(reviews)}")

        st.write("### Sek moas, lagi proses")

        analyze_and_visualize_category(reviews, "Performance", performance_keywords, threshold, similarity_measure)
        analyze_and_visualize_category(reviews, "Information", information_keywords, threshold, similarity_measure)
        analyze_and_visualize_category(reviews, "Economy", economy_keywords, threshold, similarity_measure)
        analyze_and_visualize_category(reviews, "Control", control_keywords, threshold, similarity_measure)
        analyze_and_visualize_category(reviews, "Efficiency", efficiency_keywords, threshold, similarity_measure)
        analyze_and_visualize_category(reviews, "Service", service_keywords, threshold, similarity_measure)


if __name__ == "__main__":
    main()
