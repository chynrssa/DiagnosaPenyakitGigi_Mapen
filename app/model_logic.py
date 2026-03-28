import pickle
import numpy as np
import os

# Lokasi model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/dental_model.pkl')

# Kita butuh kamus data ini agar tetap konsisten
GEJALA_DICT = {
    'G01': 'Gigi ngilu', 'G02': 'Gigi berdenyut', 'G03': 'Gigi goyang',
    'G04': 'Gigi baru muncul, gigi lama masih ada', 'G05': 'Gusi bengkak',
    'G06': 'Gigi berlubang', 'G07': 'Pipi bengkak dan terasa hangat',
    'G08': 'Nyeri saat mengunyah', 'G09': 'Sariawan', 'G10': 'Sakit gigi bungsu',
    'G11': 'Gigi berlubang tanpa sakit', 'G12': 'Gigi sakit saat diketuk',
    'G13': 'Radang', 'G14': 'Karang gigi'
}

PENYAKIT_DICT = {
    'P01': {
        'nama': 'Pulpitis Irreversible',
        'deskripsi': 'Peradangan permanen pada jaringan pulpa (saraf dan pembuluh darah di dalam gigi) yang sudah tidak bisa sembuh sendiri. Terjadi akibat karies dalam, tambalan bocor, atau trauma. Ciri khasnya adalah nyeri spontan yang berdenyut, terutama malam hari, dan rasa sakit bertahan lama bahkan setelah rangsangan dihilangkan. Satu-satunya penanganan adalah perawatan saluran akar (PSA) atau pencabutan gigi.'
    },
    'P02': {
        'nama': 'Pulpitis Reversible',
        'deskripsi': 'Peradangan ringan pada pulpa yang masih dapat pulih jika penyebabnya segera diatasi. Nyeri hanya muncul saat ada rangsangan seperti makanan/minuman dingin atau manis, lalu langsung hilang begitu rangsangan disingkirkan. Penyebab umumnya karies dangkal atau gigi retak ringan. Penanganan cukup dengan membuang karies dan menambal gigi.'
    },
    'P03': {
        'nama': 'Periodontitis',
        'deskripsi': 'Infeksi serius pada jaringan penyangga gigi (gusi, ligamen periodontal, dan tulang alveolar) yang disebabkan oleh plak dan kalkulus yang menumpuk. Ditandai dengan gusi merah, berdarah, membentuk kantung (poket), dan bau mulut. Jika dibiarkan, tulang penyangga gigi akan hancur secara bertahap hingga gigi goyang dan tanggal. Penanganan: scaling, root planing, hingga operasi periodontal.'
    },
    'P04': {
        'nama': 'Cellulitis and Abscess of Mouth',
        'deskripsi': 'Infeksi bakteri serius pada jaringan lunak rongga mulut. Abses bersifat terlokalisir (kantung nanah), sedangkan selulitis menyebar difus ke jaringan sekitarnya. Gejalanya meliputi pembengkakan wajah, nyeri hebat, demam, dan sulit membuka mulut. Sangat berbahaya karena bisa menyebar ke leher dan menyumbat jalan napas atau Ludwigs Angina. Penanganan: insisi drainase, antibiotik, dan eliminasi sumber infeksi.'
    },
    'P05': {
        'nama': 'Periapical Abscess Without Sinus',
        'deskripsi': 'Kantung nanah di ujung akar gigi akibat infeksi bakteri dari pulpa yang sudah mati, tanpa saluran pembuangan nanah ke permukaan. Gejala khas: nyeri berdenyut hebat, gigi terasa "memanjang", sangat sensitif saat digigit, serta pembengkakan lokal pada gusi atau wajah. Karena tidak ada saluran drainase, tekanan nanah menumpuk sehingga nyeri terasa lebih intens. Penanganan: PSA, insisi, atau pencabutan.'
    },
    'P06': {
        'nama': 'Caries Limited to Enamel',
        'deskripsi': 'Tahap paling awal karies gigi yang baru menyerang lapisan email (lapisan terkeras gigi) dan belum mencapai dentin. Belum menimbulkan rasa sakit karena email tidak mengandung saraf. Secara klinis tampak sebagai bercak putih kapur (white spot) atau noda cokelat di permukaan gigi. Inilah satu-satunya tahap karies yang masih bisa dipulihkan tanpa bur melalui aplikasi fluoride (remineralisasi), atau ditambal jika sudah ada kavitas.'
    },
    'P07': {
        'nama': 'Persistensi',
        'deskripsi': 'Kondisi di mana gigi sulung (gigi anak) tidak tanggal pada waktunya, padahal gigi permanen penggantinya sudah mulai tumbuh di bawahnya. Akibatnya, gigi permanen tumbuh menyimpang dari posisi normal karena jalurnya terhalangi. Sering terjadi pada gigi depan bawah. Penyebab: akar gigi sulung tidak meresorpsi sempurna. Penanganan wajib dilakukan segera: mencabut gigi sulung agar gigi permanen dapat bergerak ke posisi yang benar.'
    },
    'P08': {
        'nama': 'Stomatitis',
        'deskripsi': 'Peradangan mukosa rongga mulut yang dapat muncul di bibir, pipi bagian dalam, lidah, atau langit-langit. Bentuk paling umum adalah sariawan (aphthous ulcer) luka bulat dengan tepi merah dan tengah putih/kuning yang sangat nyeri. Penyebabnya beragam: stres, defisiensi vitamin B12/zat besi/asam folat, trauma, infeksi virus (herpes), atau imun lemah. Umumnya sembuh sendiri dalam 7–14 hari, namun kasus berat memerlukan obat kumur antiseptik atau kortikosteroid topikal.'
    },
    'P09': {
        'nama': 'Impaksi',
        'deskripsi': 'Kondisi gigi gagal erupsi sempurna karena terhalang gigi tetangga, tulang, atau jaringan lunak. Paling sering terjadi pada gigi geraham bungsu (M3/wisdom tooth). Dapat menyebabkan nyeri rahang belakang, infeksi gusi di atasnya (perikoronitis), kerusakan gigi molar di sebelahnya, hingga pembentukan kista. Impaksi diklasifikasikan berdasarkan arah (mesioangular, vertikal, horizontal, distoangular). Penanganan definitif: odontektomi (bedah pengangkatan gigi).'
    },
    'P10': {
        'nama': 'Acute Apical Periodontitis of Pulpal Origin',
        'deskripsi': 'Peradangan akut di jaringan sekitar ujung akar gigi yang dipicu oleh iritasi dari dalam saluran akar (pulpa terinfeksi atau nekrotik). Berbeda dengan periodontitis kronis yang tanpa gejala jelas, kondisi ini ditandai dengan nyeri tajam dan intens saat menggigit atau diperkusi (diketuk). Merupakan tahap transisi sebelum berkembang menjadi abses periapeks. Penanganan: PSA darurat untuk menghilangkan sumber infeksi.'
    },
    'P11': {
        'nama': 'Necrosis of Pulp',
        'deskripsi': 'Kematian total jaringan pulpa gigi akibat infeksi bakteri lanjutan atau trauma yang memutus suplai darah. Pulpa yang mati tidak merespons tes dingin maupun listrik (non-vital). Gigi sering tampak berubah warna menjadi abu-abu atau kekuningan. Meskipun tidak selalu nyeri, pulpa nekrotik menjadi sumber bakteri yang dapat memicu abses atau kista apeks. Penanganan: PSA untuk membersihkan saluran akar dan mencegah penyebaran infeksi.'
    },
    'P12': {
        'nama': 'Gingivitis Kronis',
        'deskripsi': 'Peradangan gusi yang berlangsung jangka panjang akibat akumulasi plak bakteri di sepanjang garis gusi. Ditandai dengan gusi merah, bengkak, mudah berdarah saat menyikat gigi, namun belum terjadi kerusakan tulang penyangga. Inilah yang membedakannya dari periodontitis. Kondisi ini masih sepenuhnya reversible jika kebersihan mulut diperbaiki. Jika dibiarkan, akan berkembang menjadi periodontitis yang merusak tulang secara permanen. Penanganan: scaling profesional dan instruksi kebersihan mulut.'
    }
}

FITUR_GEJALA = list(GEJALA_DICT.keys())


def load_model():
    """Memuat model yang sudah dilatih"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None


def prediksi_penyakit(list_gejala_input):
    """
    Menerima list kode gejala ['G01', 'G03']
    Mengembalikan kode penyakit, nama penyakit, dan probabilitas
    """
    model = load_model()
    if not model:
        return "Error", "Model tidak ditemukan", []

    # Konversi input user ke vektor biner (0/1)
    input_vector = [0] * len(FITUR_GEJALA)
    for g in list_gejala_input:
        if g in FITUR_GEJALA:
            idx = FITUR_GEJALA.index(g)
            input_vector[idx] = 1

    input_array = np.array([input_vector])

    # Prediksi
    prediksi_kode = model.predict(input_array)[0]
    probs = model.predict_proba(input_array)[0]

    # 🔥 FIX 1: Ambil hanya nama (bukan dict)
    data_penyakit = PENYAKIT_DICT.get(prediksi_kode, None)

    if data_penyakit:
        nama_penyakit = data_penyakit['nama']
        deskripsi = data_penyakit['deskripsi']
    else:
        nama_penyakit = "Tidak Diketahui"
        deskripsi = "-"

    # 🔥 FIX 2: Pastikan nama di grafik berupa string
    classes = model.classes_
    hasil_probs = []
    for i, cls in enumerate(classes):
        hasil_probs.append({
            'kode': cls,
            'nama': PENYAKIT_DICT.get(cls, {}).get('nama', cls),
            'probabilitas': round(probs[i], 4)
        })

    return prediksi_kode, nama_penyakit, deskripsi, hasil_probs