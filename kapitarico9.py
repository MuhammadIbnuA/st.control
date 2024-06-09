from google_play_scraper import Sort, reviews
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import googletrans
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import xlsxwriter

def ambil_ulasan_terbatas(app_id, bahasa='id', negara='id', urut=Sort.NEWEST, filter_nilai=""):
    semua_ulasan_konten = []
    ulasan_id_terkumpul = set()

    hasil, kelanjutan_token = reviews(app_id, lang=bahasa, country=negara, sort=urut, count=200, filter_score_with=filter_nilai)
    
    for ulasan in hasil:
        if ulasan['reviewId'] not in ulasan_id_terkumpul:
            semua_ulasan_konten.append(ulasan['content'])
            ulasan_id_terkumpul.add(ulasan['reviewId'])

    return semua_ulasan_konten

def normalisasi_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'[\U00010000-\U0010ffff]', '', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = re.sub(r'[^\w\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def saring_ulasan_cosine_similarity(ulasan, kata_kunci, threshold=0.05):
    tfidf_vectorizer = TfidfVectorizer()
    kata_kunci_vektor = tfidf_vectorizer.fit_transform([" ".join(kata_kunci)])
    
    ulasan_tersaring = []
    for teks_ulasan in ulasan:
        ulasan_vektor = tfidf_vectorizer.transform([teks_ulasan])
        similarity = cosine_similarity(ulasan_vektor, kata_kunci_vektor)
        if similarity >= threshold:
            ulasan_tersaring.append(teks_ulasan)
    
    return ulasan_tersaring

def terjemahkan_ulasan(ulasan):
    translator = Translator()
    ulasan_diterjemahkan = []
    for teks in ulasan:
        try:
            translated = translator.translate(teks, src='id', dest='en').text
            ulasan_diterjemahkan.append(translated)
        except Exception as e:
            ulasan_diterjemahkan.append(teks)  # Jika gagal diterjemahkan, gunakan teks asli
    return ulasan_diterjemahkan

def analisis_sentimen(ulasan):
    analyzer = SentimentIntensityAnalyzer()
    skor_sentimen = [analyzer.polarity_scores(teks)["compound"] for teks in ulasan]
    return skor_sentimen

def kategorikan_sentimen(skor):
    kategori = []
    for score in skor:
        if score <= -0.6:
            kategori.append('Sangat Kurang Puas')
        elif score <= -0.2:
            kategori.append('Kurang Puas')
        elif score <= 0.2:
            kategori.append('Cukup')
        elif score <= 0.6:
            kategori.append('Puas')
        else:
            kategori.append('Sangat Puas')
    return kategori

def main():
    print("Filter Ulasan")
    
    app_id = input("Masukkan ID aplikasi id.or.muhammadiyah.quran: ")

    if app_id:
        ulasan_konten = ambil_ulasan_terbatas(app_id)

        ulasan_konten_normalisasi = [normalisasi_teks(ulasan) for ulasan in ulasan_konten]

        kata_kunci = ["rekaman", "informasi", "detail", "fakta", "angka", "statistik", "sistem", "struktur", "kerangka", "organisasi", "pengaturan", "metode", "informasi", "intelijen", "pengetahuan", "berita", "pembaruan", "wawasan", "keamanan", "keselamatan", "perlindungan", "keamanan", "pertahanan", "pengamanan", "akses", "jalan masuk", "jangkauan", "pendekatan", "koneksi", "penerimaan", "terjamin", "dijamin", "diyakinkan", "aman", "pasti", "dikonfirmasi", "penggunaan", "penggunaan", "aplikasi", "pekerjaan", "pemanfaatan", "implementasi", "error", "kesalahan", "kegagalan", "cacat", "kekurangan", "ketidakakuratan", "terlindungi", "dijaga", "dilindungi", "dipelihara", "dipertahankan", "diamankan", "diakses", "diakses", "diperoleh", "diambil", "dikumpulkan", "mudah", "sederhana", "mudah dilakukan", "langsung", "tidak rumit", "nyaman", "konsumen", "pelanggan", "pengguna", "pembeli", "pemborong", "menggunakan", "memanfaatkan", "menggunakan", "menerapkan", "menjalankan", "mengoperasikan", "aman", "aman", "terjamin", "terlindungi", "terpercaya", "hak", "hak", "hak akses", "wewenang", "kuasa", "prerogatif", "batasan", "batasan", "pembatasan", "kendala", "garis batas", "pedoman", "kecurangan", "penipuan", "pengelabuan", "tipu daya", "ketidakjujuran", "pelanggaran", "pribadi", "personal", "pribadi", "rahasia", "sensitif", "intim", "berbagai", "beragam", "bervariasi", "beberapa", "berbagai macam", "bermacam-macam", "kejahatan", "kejahatan", "pelanggaran", "kesalahan", "pelanggaran ringan", "pelanggaran"]
        
        ulasan_dengan_kata_kunci = saring_ulasan_cosine_similarity(ulasan_konten_normalisasi, kata_kunci)

        ulasan_diterjemahkan = terjemahkan_ulasan(ulasan_dengan_kata_kunci)

        skor_sentimen = analisis_sentimen(ulasan_diterjemahkan)

        kategori_sentimen = kategorikan_sentimen(skor_sentimen)

        df_ulasan_dengan_kata_kunci = pd.DataFrame({
            "Nomor Ulasan": range(1, len(ulasan_dengan_kata_kunci) + 1),
            "Ulasan Asli": ulasan_dengan_kata_kunci,
            "Ulasan Diterjemahkan": ulasan_diterjemahkan,
            "Skor Sentimen": skor_sentimen,
            "Kategori Sentimen": kategori_sentimen
        })

        # Membuat pie chart untuk persebaran kategori sentimen
        kategori_counts = df_ulasan_dengan_kata_kunci["Kategori Sentimen"].value_counts()

        plt.figure(figsize=(10, 7))
        plt.pie(kategori_counts, labels=kategori_counts.index, autopct='%1.1f%%', startangle=140, colors=['red', 'orange', 'yellow', 'green', 'blue'])
        plt.axis('equal')
        plt.title('Persebaran Kategori Sentimen')
        
        # Simpan pie chart ke buffer
        plt.savefig("pie_chart_sentimen.png")
        plt.show()

        # Menyimpan data dan pie chart ke file Excel
        writer = pd.ExcelWriter("ulasan_tersaring_dan_sentimen.xlsx", engine='xlsxwriter')
        df_ulasan_dengan_kata_kunci.to_excel(writer, sheet_name='Ulasan dan Sentimen', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Ulasan dan Sentimen']

        # Menambahkan pie chart ke worksheet
        worksheet.insert_image('G2', 'pie_chart_sentimen.png')

        writer.close()

        print("Ulasan yang mengandung kata kunci dan analisis sentimen serta pie chart telah disimpan ke ulasan_tersaring_dan_sentimen.xlsx")

if __name__ == "__main__":
    main()
