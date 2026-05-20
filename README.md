# 🧠 Çok Modlu Karşıtlık Füzyonu ile Güven Durumu Temsil Öğrenmesi

> **Confidence-State Representation Learning via Multimodal Contrastive Fusion**  
> EEG + Periferik Biyosinyaller + Davranışsal Performans

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-green.svg)](https://streamlit.io/)
[![Lisans](https://img.shields.io/badge/Lisans-MIT-yellow.svg)](LICENSE)

---

## 📋 Proje Hakkında

Bu proje, eş zamanlı kaydedilen aşağıdaki verilerden güven durumunun otomatik olarak sınıflandırılmasına yönelik **çok modlu bir derin öğrenme çerçevesi** sunmaktadır:

- 🧠 **EEG** band güç özellikleri (10 kanal, NeuroSky)
- 💧 **Periferik biyosinyaller** — EDA, BVP, deri sıcaklığı, ivmeölçer (Empatica E4)
- 📊 **Davranışsal performans** — oyun skorları + psikometrik ölçekler (Rosenberg, GSE)

Temel katkı, **InfoNCE karşıtlık kaybı** aracılığıyla EEG ve periferik gömmeleri ortak bir gizil uzayda hizalayan, çapraz dikkat mekanizması ve ayrılmış bir BehavioralEncoder ile zenginleştirilmiş **karşıtlık füzyon mimarisidir**.

| Ölçüt | Değer |
|-------|-------|
| Test Doğruluğu | **%70,7** (rastgeleden +37,4 puan) |
| Test Makro F1 | **0,702** |
| LOSO Ortalama F1 | 0,444 ± 0,256 (33 kıvrım) |
| DEAP Sıfır-Geçiş | %41,5 doğruluk |
| Parametre Sayısı | 251.475 |

---

## 🏗️ Mimari

```
EEG (10×512)          Periferik (6×512)       Davranışsal (9,)
     │                       │                      │
  EEGNet                  1B-ESA              BehavioralEncoder
  (128-d)                 (128-d)                  (32-d)
     │                       │                      │
     └──────── Çapraz Dikkat + InfoNCE ─────────────┘
                             │
                    Birleştirme (288-d)
                             │
                      MLP → 3 Sınıf
                  (Nötr / Pozitif / Negatif)
```

**Kayıp fonksiyonu:** `L = 0,5 × L_InfoNCE + 0,5 × L_CrossEntropy`

---

## 📁 Proje Yapısı

```
confidence_eeg_project/
├── src/
│   ├── data/
│   │   ├── cosubio_loader.py        # CoSuBio veri yükleyici
│   │   ├── behavioral_loader.py     # Davranışsal veri işleme
│   │   └── build_cosubio_dataset.py # Veri seti oluşturma
│   ├── models/
│   │   ├── model.py                 # Ana model (ConfidenceEEGModel)
│   │   ├── eeg_encoder.py           # EEGNet kodlayıcı
│   │   ├── peripheral_encoder.py    # 1B-ESA periferik kodlayıcı
│   │   ├── behavioral_encoder.py    # MLP davranışsal kodlayıcı
│   │   └── contrastive_fusion.py    # InfoNCE + Çapraz Dikkat
│   └── training/
│       ├── trainer.py               # Model eğitimi
│       ├── dataset.py               # PyTorch veri seti
│       ├── loso.py                  # LOSO çapraz doğrulama
│       └── deap_cross_dataset.py    # DEAP sıfır-geçiş testi
├── data/
│   ├── raw/cosubio/                 # Ham CoSuBio verisi (dahil değil)
│   ├── raw/deap/                    # Ham DEAP verisi (dahil değil)
│   └── processed_cosubio/           # İşlenmiş numpy dizileri (dahil değil)
├── experiments/
│   └── checkpoints_cosubio/         # Model ağırlıkları (dahil değil)
├── results/
│   ├── loso_results.json            # LOSO sonuçları
│   └── deap_cross_dataset_results.json
├── ui/
│   └── app.py                       # Streamlit dashboard
├── notebooks/                       # Keşif notları
├── requirements.txt
└── README.md
```

---

## 🚀 Hızlı Başlangıç

### 1. Ortam Kurulumu

```bash
# Conda ortamı (GPU — eğitim için)
conda create -n confidence_eeg python=3.10
conda activate confidence_eeg
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Sanal ortam (CPU — dashboard için)
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Veri Hazırlama

```bash
cd src/data
python build_cosubio_dataset.py
# → data/processed_cosubio/{eeg,peripheral,labels,behavioral,subject_ids}.npy
```

### 3. Model Eğitimi

```bash
cd src/training
python trainer.py
# Kontrol noktası: experiments/checkpoints_cosubio/best_model.pt
```

### 4. LOSO Çapraz Doğrulama

```bash
cd src/training
python loso.py
# → results/loso_results.json
```

### 5. DEAP Sıfır-Geçiş Testi

```bash
cd src/training
python deap_cross_dataset.py
# → results/deap_cross_dataset_results.json
```

### 6. Dashboard'u Başlat

```bash
streamlit run ui/app.py
# → localhost:8501
```

---

## 📊 Sonuçlar

### Ana Sonuçlar (CoSuBio)

| Bölünme | Doğruluk | Makro F1 |
|---------|----------|----------|
| Eğitim | %69,4 | 0,688 |
| Doğrulama | %70,0 | 0,694 |
| **Test** | **%70,7** | **0,702** |

Rastgele taban çizgisi: %33,3 → Model: **+37,4 puan**

### Ablasyon Çalışması

| Yapılandırma | Doğruluk | Makro F1 |
|-------------|----------|----------|
| Yalnızca EEG | %37,7 | 0,206 |
| Yalnızca Periferik | %59,1 | 0,574 |
| Füzyon (InfoNCE olmadan) | %66,3 | 0,651 |
| **Tam Model** | **%70,7** | **0,702** |

### LOSO Çapraz Doğrulama (33 kıvrım)

| Ölçüt | Ortalama | Std. Sapma | Medyan |
|-------|----------|------------|--------|
| Doğruluk | %57,1 | ±%21,6 | — |
| Makro F1 | 0,444 | ±0,256 | 0,528 |
| F1 ≥ 0,5 kıvrım | 18/33 | — | — |

### DEAP'e Sıfır-Geçiş Transferi

| | CoSuBio | DEAP (0-geçiş) | Rastgele |
|---|---------|----------------|----------|
| Doğruluk | %70,7 | %41,5 | %33,3 |
| Makro F1 | 0,702 | 0,325 | 0,333 |

---

## ⚙️ Eğitim Hiperparametreleri

| Parametre | Değer |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate| 3×10⁻⁴ |
| Batch Size | 64 |
| Epochs | 50 (en iyi @ 38) |
| Scheduler | Kosinüs Tavlaması |
| Weight Decay | 1×10⁻⁴ |
| α (loss weight) | 0,5 |
| τ (temperature) | 0,07 |

---

## 📦 Veri Setleri

### CoSuBio (Birincil)
- 34 katılımcı, 3 güven koşulu (Nötr / Pozitif / Negatif)
- EEG: 10 band gücü özelliği @ 1 Hz (NeuroSky)
- Periferik: EDA, BVP, sıcaklık, ivmeölçer @ 4–64 Hz (Empatica E4)
- Davranışsal: 7 oyun skoru + Rosenberg Öz-Saygı + Genel Öz-Yeterlik
- 63.974 epoch (4 saniyelik pencereler, %50 örtüşme)
- Yayın: Data in Brief, 2026

### DEAP (Çapraz Veri Seti)
- 32 katılımcı, 40 müzik videosu
- EEG: 32 kanal @ 128 Hz
- Periferik: EDA, BVP, EMG, sıcaklık, EOG
- Etiketler: Valence / Arousal / Dominance
- Yalnızca sıfır-geçiş transfer değerlendirmesi için kullanıldı

> ⚠️ **Not:** Ham veri dosyaları boyut ve lisans kısıtları nedeniyle bu depoya dahil edilmemiştir. Lütfen resmi kaynaklardan indirin.

---

## 🔧 Temel Bağımlılıklar

```
torch>=2.0.0
streamlit>=1.28.0
plotly>=5.17.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
pandas>=2.0.0
wandb>=0.16.0
```
