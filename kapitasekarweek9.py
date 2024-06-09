import googletrans
from googletrans import Translator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
from google_play_scraper import Sort, reviews

def ambil_ulasan(id_aplikasi, jumlah=200, bahasa='id', negara='id', urut=Sort.NEWEST, filter_skor=""):
    semua_isi_ulasan = []
    daftar_id_ulasan_terkumpul = set()

    hasil, _ = reviews(id_aplikasi, lang=bahasa, country=negara, sort=urut, count=jumlah, filter_score_with=filter_skor)
    
    for ulasan in hasil:
        if ulasan['reviewId'] not in daftar_id_ulasan_terkumpul:
            # Normalisasi teks
            teks_normal = normalisasi_teks(ulasan['content'])
            semua_isi_ulasan.append(teks_normal)
            daftar_id_ulasan_terkumpul.add(ulasan['reviewId'])

            if len(semua_isi_ulasan) >= jumlah:
                break  # Menghentikan loop jika sudah mencapai batas jumlah ulasan yang diinginkan

    return semua_isi_ulasan

def normalisasi_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'[\U00010000-\U0010ffff]', '', teks)  # Hapus karakter non-BMP
    teks = re.sub(r'\d+', '', teks)  # Hapus angka
    teks = re.sub(r'[^\w\s]', '', teks)  # Hapus tanda baca
    teks = re.sub(r'\s+', ' ', teks).strip()  # Hapus spasi berlebih
    return teks

def filter_ulasan_berdasarkan_kata_kunci(ulasan, kata_kunci):
    ulasan_terfilter = []
    for review in ulasan:
        if any(re.search(r'\b{}\b'.format(re.escape(keyword)), review) for keyword in kata_kunci):
            ulasan_terfilter.append(review)
    return ulasan_terfilter

def translate_reviews(ulasan, src='id', dest='en'):
    translator = Translator()
    translated_reviews = []
    for review in ulasan:
        translated = translator.translate(review, src=src, dest=dest)
        translated_reviews.append(translated.text)
    return translated_reviews

def sentiment_analysis_vader(reviews):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for review in reviews:
        score = analyzer.polarity_scores(review)['compound']
        sentiments.append(score)
    return sentiments

def sentiment_analysis_textblob(reviews):
    sentiments = []
    for review in reviews:
        score = TextBlob(review).sentiment.polarity
        sentiments.append(score)
    return sentiments

def categorize_sentiments(scores):
    categories = []
    for score in scores:
        if score <= -0.6:
            categories.append('Sangat Kurang Puas')
        elif score <= -0.2:
            categories.append('Kurang Puas')
        elif score <= 0.2:
            categories.append('Cukup')
        elif score <= 0.6:
            categories.append('Puas')
        else:
            categories.append('Sangat Puas')
    return categories

def main():
    id_aplikasi = input("Masukkan ID Aplikasi di Google Play Store: ")

    if id_aplikasi:
        jumlah = 200  # Mengatur jumlah ulasan yang akan diambil

        isi_ulasan = ambil_ulasan(id_aplikasi, jumlah)

        kata_kunci = [
            # kata kunci satu kata
            "penggunaan", "ketersediaan", "menggunakan", "mengevaluasi", "memperhatikan", "mudah", "cepat", "meningkatkan", 
            "diakses", "baik", "sesuai", "dipahami", "digunakan", "pengguna", "dijalankan", "secara", "stabil", "merespon", 
            "tampilan", "memuaskan", "tepat", "hasil", "kepuasan", "efisien",
            # kata kunci dua kata
            "mudah diakses", "mudah dipahami", "meningkatkan kepuasan", "cepat mudah", "tepat waktu", "waktu hasil", 
            "dipahami mudah", "mudah digunakan", "digunakan mudah", "baik sesuai",
            # kata kunci tiga kata atau lebih
            "meningkatkan produktivitas",
            # hasil pengembangan kata kunci
            "penggunaan", "pemanfaatan", "ketersediaan", "aksesibilitas", "menggunakan", "evaluasi", "penilaian", "memperhatikan", 
            "perhatian", "pertimbangan", "kemudahan", "keterjangkauan", "kecepatan", "kelancaran", "cepatnya", "peningkatan", 
            "perbaikan", "peningkatan kualitas", "akses mudah", "kualitas baik", "kebaikan", "keunggulan", "keterbacaan", 
            "kejelasan", "pemahaman yang baik", "relevan", "cocok", "relevansi", "punctuality", "ketepatan waktu", 
            "waktu yang diperlukan untuk hasil", "pemahaman", "user", "pemakai", "user-friendly", "aplikasi yang mudah digunakan", 
            "penggunaan yang mudah", "operasional", "dengan", "melalui", "secara efektif", "kestabilan", "keberlangsungan", 
            "konsistensi", "responsif", "respons", "tanggapan", "antarmuka", "visualisasi", "kepuasan pengguna", "kepuasan pelanggan", 
            "konsistensi", "kesenjangan yang baik", "peningkatan efisiensi", "kinerja yang lebih baik", "akurat", "keakuratan", 
            "ketepatan", "outcome", "output", "efisiensi", "penggunaan yang efisien"
        ]

        ulasan_dengan_kata_kunci = filter_ulasan_berdasarkan_kata_kunci(isi_ulasan, kata_kunci)
        translated_reviews = translate_reviews(ulasan_dengan_kata_kunci)

        vader_sentiments = sentiment_analysis_vader(translated_reviews)
        textblob_sentiments = sentiment_analysis_textblob(translated_reviews)

        vader_categories = categorize_sentiments(vader_sentiments)
        textblob_categories = categorize_sentiments(textblob_sentiments)

        df_ulasan_dengan_kata_kunci = pd.DataFrame({
            "Nomor Ulasan": range(1, len(ulasan_dengan_kata_kunci) + 1),
            "Ulasan": ulasan_dengan_kata_kunci,
            "Ulasan Terjemahan": translated_reviews,
            "Sentimen VADER": vader_sentiments,
            "Kategori VADER": vader_categories,
            "Sentimen TextBlob": textblob_sentiments,
            "Kategori TextBlob": textblob_categories
        })

        print("Ulasan yang mengandung kata kunci dan hasil analisis sentimen:")
        print(df_ulasan_dengan_kata_kunci)

if __name__ == "__main__":
    main()

