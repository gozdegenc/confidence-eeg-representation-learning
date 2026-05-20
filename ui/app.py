
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import time

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src" / "models"))
sys.path.append(str(ROOT / "src" / "training"))

# ── Renkler ───────────────────────────────────────────────────────────
C = {
    "teal":    "#1D9E75", "teal_l":  "#E1F5EE", "teal_d":  "#085041",
    "blue":    "#185FA5", "blue_l":  "#E6F1FB", "blue_d":  "#0C447C",
    "purple":  "#534AB7", "purple_l":"#EEEDFE", "purple_d":"#3C3489",
    "amber":   "#BA7517", "amber_l": "#FAEEDA", "amber_d": "#633806",
    "red":     "#E24B4A", "red_l":   "#FCEBEB", "red_d":   "#791F1F",
    "gray":    "#888780", "gray_l":  "#F1EFE8",
    "coral":   "#D85A30",
}

ABLATION = {
    "EEG Only":          {"acc":0.377,"f1":0.206,"color":"#534AB7","desc":"Sadece beyin dalgaları"},
    "Peripheral Only":   {"acc":0.591,"f1":0.574,"color":"#185FA5","desc":"Sadece beden sinyalleri"},
    "Fusion (no CL)":    {"acc":0.663,"f1":0.651,"color":"#BA7517","desc":"İki encoder · InfoNCE yok"},
    "Full Model (ours)": {"acc":0.707,"f1":0.702,"color":"#1D9E75","desc":"EEG + Periferik + InfoNCE ✓"},
}

TRAIN_LOSS=[2.6557,2.5858,2.5556,2.5421,2.5279,2.5149,2.5052,2.4970,2.4898,2.4827,
            2.4745,2.4670,2.4601,2.4564,2.4504,2.4413,2.4367,2.4313,2.4265,2.4210,
            2.4160,2.4128,2.4052,2.4012,2.3971,2.3938,2.3899,2.3856,2.3834,2.3803,
            2.3760,2.3720,2.3696,2.3664,2.3613,2.3609,2.3613,2.3584,2.3548,2.3522,
            2.3506,2.3489,2.3503,2.3459,2.3469,2.3466,2.3446,2.3443,2.3444,2.3434]
VAL_LOSS =[2.6211,2.5578,2.5481,2.5255,2.5196,2.5114,2.4905,2.4927,2.4862,2.4653,
            2.4645,2.4532,2.4362,2.4425,2.4277,2.4146,2.4189,2.4197,2.4042,2.3944,
            2.3886,2.3919,2.3883,2.3747,2.3882,2.3672,2.3638,2.3619,2.3528,2.3585,
            2.3482,2.3429,2.3520,2.3484,2.3402,2.3410,2.3473,2.3359,2.3391,2.3384,
            2.3402,2.3349,2.3337,2.3319,2.3358,2.3341,2.3379,2.3340,2.3371,2.3386]
VAL_F1  =[0.409,0.487,0.494,0.537,0.536,0.543,0.562,0.562,0.572,0.593,
           0.594,0.612,0.618,0.620,0.635,0.636,0.632,0.628,0.645,0.657,
           0.655,0.653,0.659,0.669,0.648,0.670,0.675,0.678,0.682,0.680,
           0.687,0.699,0.682,0.684,0.694,0.697,0.688,0.703,0.696,0.691,
           0.692,0.699,0.696,0.700,0.694,0.697,0.693,0.699,0.691,0.694]

PAGES = ["🏠 Genel Bakış","⚡ Canlı Demo","🔬 Model","📈 Eğitim","🗂 Veri","ℹ Hakkında"]

# ── Sayfa config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Confidence-State EEG", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
[data-testid="stSidebar"]{display:none!important}
[data-testid="collapsedControl"]{display:none!important}
[data-testid="stAppViewContainer"]{background:#F8FAFC!important}
.main .block-container{padding-top:0!important;max-width:1180px!important;
  padding-left:2rem!important;padding-right:2rem!important}

/* Metrik kart */
.mc{border-radius:12px;padding:18px 20px;margin-bottom:4px}
.mc-lbl{font-size:12px;font-weight:600;margin-bottom:6px}
.mc-val{font-size:32px;font-weight:800;line-height:1}
.mc-sub{font-size:12px;margin-top:5px;opacity:0.75}

/* Bölüm */
.sec{font-size:20px;font-weight:800;color:#1A2E4A;margin-bottom:4px}
.sec-s{font-size:13px;color:#64748B;margin-bottom:18px}

/* Adım kartı */
.step{display:flex;gap:16px;align-items:flex-start;
  background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;
  padding:18px 22px;margin-bottom:10px}
.step-n{font-size:30px;font-weight:900;line-height:1;flex-shrink:0;opacity:.35}
.step-t{font-size:14px;font-weight:700;color:#1A2E4A;margin-bottom:4px}
.step-d{font-size:12px;color:#64748B;line-height:1.7}

/* Ablasyon kart */
.abl{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
  padding:14px 16px;margin-bottom:8px}
.abl.best{border-color:#1D9E75;background:#E1F5EE}
.abl-n{font-size:13px;font-weight:700;color:#1A2E4A}
.abl-n.best{color:#085041}
.abl-d{font-size:11px;color:#64748B;margin:3px 0 10px}
.abl-d.best{color:#0F6E56}
.bar-bg{height:10px;background:#E2E8F0;border-radius:5px;margin-bottom:6px}
.abl-sc{display:flex;justify-content:space-between;font-size:11px;color:#64748B}

/* Pred */
.pred-box{border-radius:12px;padding:18px 20px;
  display:flex;align-items:center;gap:16px;margin-bottom:14px}
.pred-cls{font-size:22px;font-weight:700}
.prob-h{display:flex;justify-content:space-between;
  font-size:12px;color:#64748B;margin-bottom:4px}
.prob-tr{height:8px;background:#E2E8F0;border-radius:4px;overflow:hidden;margin-bottom:10px}
.prob-f{height:100%;border-radius:4px}

/* Embed grid */
.egrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.ebox{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px}
.elbl{font-size:10px;color:#64748B}
.eval{font-size:16px;font-weight:700;color:#1A2E4A;margin-top:2px}

/* Sinyal chip */
.chip{border-radius:8px;padding:9px 12px;margin-bottom:5px}
.chip-n{font-size:11px;font-weight:600;margin-bottom:1px}
.chip-b{font-size:10px}

/* Mimari şerit */
.arc{display:flex;border-radius:12px;overflow:hidden;
  border:1px solid #E2E8F0;margin-bottom:20px;background:#FFFFFF}
.arc-s{flex:1;padding:12px 14px;text-align:center}
.arc-s:not(:last-child){border-right:1px solid #E2E8F0}
.arc-sl{font-size:12px;font-weight:600}
.arc-ss{font-size:10px;color:#64748B;margin-top:3px}
.arc-a{width:26px;display:flex;align-items:center;justify-content:center;
  background:#F8FAFC;color:#94A3B8;font-size:15px;flex-shrink:0}

.div{border:none;border-top:1px solid #E2E8F0;margin:24px 0}
.hero{background:linear-gradient(135deg,#1A2E4A,#0D5C3E);border-radius:16px;
  padding:38px 44px;margin:18px 0 24px;position:relative;overflow:hidden}
.hero::after{content:'';position:absolute;top:-80px;right:-80px;
  width:340px;height:340px;background:radial-gradient(circle,rgba(29,158,117,.18),transparent 65%);
  border-radius:50%}
.hero-tag{font-size:11px;font-weight:700;letter-spacing:3px;color:#3ECFB2;margin-bottom:12px}
.hero-title{font-size:28px;font-weight:800;color:#FFF;line-height:1.25;margin-bottom:10px}
.hero-sub{font-size:14px;color:#A8C4E0;line-height:1.65;max-width:560px;margin-bottom:26px}
.hero-stats{display:flex;gap:32px;flex-wrap:wrap}
.hst-v{font-size:34px;font-weight:800;line-height:1}
.hst-l{font-size:11px;color:#7BA3CC;margin-top:4px}

/* Butonlar */
.stButton>button[kind=primary]{background:#1D9E75!important;
  border-color:#1D9E75!important;color:#fff!important;
  font-weight:600!important;border-radius:8px!important}
.stButton>button[kind=secondary]{border-color:#E2E8F0!important;border-radius:8px!important}
[data-testid="stRadio"] label{color:#1A2E4A!important}
            
</style>
""", unsafe_allow_html=True)

# ── Model yükle ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from model import ConfidenceEEGModel
    ckpt = ROOT / "experiments" / "checkpoints_cosubio" / "best_model.pt"
    m = ConfidenceEEGModel(n_eeg_channels=10, n_periph_channels=6, n_samples=512)
    m.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
    m.eval()
    return m

mdl, model_ok = None, False
try:
    mdl = load_model(); model_ok = True
except Exception:
    pass

def run_inf(model, eeg, periph):
    et = torch.from_numpy(eeg).float().unsqueeze(0)
    pt = torch.from_numpy(periph).float().unsqueeze(0)
    bt = torch.zeros(1, 9)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(et, pt, bt)
    ms  = (time.perf_counter()-t0)*1000
    prb = F.softmax(out["logits"][0], dim=-1).numpy()
    ze  = out["z_eeg"][0].numpy()
    zp  = out["z_periph"][0].numpy()
    return {"pred":int(prb.argmax()),"probs":prb,"z_eeg":ze,"z_periph":zp,
            "cos_sim":float(F.cosine_similarity(out["z_eeg"],out["z_periph"]).item()),
            "eeg_norm":float(np.linalg.norm(ze)),"periph_norm":float(np.linalg.norm(zp)),"ms":ms}

def make_sig(lv, noise):
    bias = {"Düşük": -2.0, "Nötr": 0.0, "Yüksek": 2.0}[lv]
    np.random.seed({"Düşük": 42, "Nötr": 0, "Yüksek": 7}[lv])
    eeg    = (np.random.randn(10, 512) * noise + bias).astype(np.float32)
    np.random.seed({"Düşük": 42, "Nötr": 0, "Yüksek": 7}[lv] + 1)
    periph = (np.random.randn(6,  512) * noise + bias).astype(np.float32)
    return eeg, periph

# ── Üst nav bar ───────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

stat_c = "#1D9E75" if model_ok else "#E24B4A"
stat_b = "#E1F5EE" if model_ok else "#FCEBEB"
stat_t = "✓ Model hazır" if model_ok else "✗ Model yüklenemedi"

# Görsel nav bar (sadece gösterim)
active_i = PAGES.index(st.session_state.page)
links_html = "".join(
    f'<div style="padding:14px 16px;font-size:13px;font-weight:500;white-space:nowrap;'
    f'color:{"#1D9E75" if i==active_i else "#64748B"};'
    f'border-bottom:2px solid {"#1D9E75" if i==active_i else "transparent"}">{p}</div>'
    for i, p in enumerate(PAGES)
)
st.markdown(f"""
<div style="position:sticky;top:0;z-index:999;background:#FFFFFF;
  border-bottom:1px solid #E2E8F0;padding:0 2rem;
  display:flex;align-items:center;gap:0;margin-bottom:0;
  box-shadow:0 1px 8px rgba(0,0,0,.06)">
  <div style="font-size:15px;font-weight:800;color:#1A2E4A;
    margin-right:28px;white-space:nowrap;padding:14px 0;display:flex;align-items:center;gap:8px">
    🧠 Confidence<span style="color:#1D9E75">EEG</span>
  </div>
  <div style="display:flex;gap:0;align-items:center;flex:1">{links_html}</div>
  <div style="margin-left:auto;font-size:11px;font-weight:600;
    color:{stat_c};background:{stat_b};padding:5px 14px;border-radius:20px;
    flex-shrink:0">{stat_t}</div>
</div>
""", unsafe_allow_html=True)

# İşlevsel butonlar (küçük, görsel nav'ın altında)
nav_cols = st.columns(len(PAGES))
for i, (col, p) in enumerate(zip(nav_cols, PAGES)):
    with col:
        if st.button(p, key=f"nb_{i}",
                     type="primary" if st.session_state.page==p else "secondary",
                     use_container_width=True):
            st.session_state.page = p
            st.rerun()

page = st.session_state.page
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# GENEL BAKIŞ
# ════════════════════════════════════════════════════════════
if page == "🏠 Genel Bakış":

    # Hero + EEG dalgası animasyonu
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">CSE0540 · DERİN ÖĞRENME · MAYIS 2026</div>
      <div class="hero-title">İnsan güven durumunu<br>fizyolojik sinyallerden öğrendik</div>
      <div class="hero-sub">
        EEG band güçleri + periferik biyosinyaller + davranışsal performans →
        <b style="color:#3ECFB2">multimodal derin öğrenme</b> → güven sınıfı tahmini
      </div>
      <div class="hero-stats">
        <div><div class="hst-v" style="color:#3ECFB2">70.7%</div>
          <div class="hst-l">Test Accuracy</div></div>
        <div><div class="hst-v" style="color:#5B8AF0">0.702</div>
          <div class="hst-l">Macro F1</div></div>
        <div><div class="hst-v" style="color:#F5A623">34</div>
          <div class="hst-l">Subject</div></div>
        <div><div class="hst-v" style="color:#E05C7A">41.5%</div>
          <div class="hst-l">DEAP Zero-shot</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Canlı EEG dalgası
    t = np.linspace(0, 8*np.pi, 400)
    wfig = go.Figure()
    for i, (amp, freq, color, name) in enumerate([
        (0.7, 1.5, "#1D9E75", "Alpha"),
        (0.5, 2.8, "#185FA5", "Beta"),
        (0.4, 0.9, "#534AB7", "Theta"),
    ]):
        y = amp*np.sin(freq*t + i) + np.cos(freq*0.7*t)*0.3*amp
        wfig.add_trace(go.Scatter(x=np.linspace(0,4,400), y=y+i*2,
            mode="lines", name=name, line=dict(color=color, width=2),
            fill="tozeroy" if i==0 else "none",
            fillcolor="rgba(29,158,117,0.06)",
        ))
    wfig.update_layout(height=130, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True, hovermode=False,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.1,
                    font=dict(size=11, color="#64748B")),
        xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(wfig, use_container_width=True,
                    config={"displayModeBar":False,"staticPlot":True})

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Neden önemli
    st.markdown('<div class="sec">Neden önemli?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Güven durumunu otomatik ölçmek üç alanda değer taşıyor</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,icon,title,desc,col_ in zip([c1,c2,c3],
        ["🎓","🏥","🤖"],
        ["Eğitim Teknolojisi","Klinik Psikoloji","İnsan-Makine"],
        ["Öğrenci sınavda gerçekten emin mi? Güven düşünce sistem anlık geri bildirim sunar.",
         "Anksiyete tedavisinde güven durumu nesnel fizyolojik veriyle takip edilir.",
         "Kullanıcı strese girdiğinde arayüz otomatik daha basit moda geçer."],
        [C["teal"],C["blue"],C["amber"]],
    ):
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;
              padding:20px;border-top:3px solid {col_};min-height:165px">
              <div style="font-size:26px;margin-bottom:9px">{icon}</div>
              <div style="font-size:13px;font-weight:700;color:{col_};margin-bottom:7px">{title}</div>
              <div style="font-size:12px;color:#64748B;line-height:1.65">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Model mimarisi görsel diyagram
    st.markdown('<div class="sec">Model Mimarisi</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">3 encoder → contrastive fusion → güven sınıfı · 251K parametre</div>', unsafe_allow_html=True)

    # Plotly diyagramı
    arch_fig = go.Figure()
    boxes = [
        (0.05, 0.6, 0.22, 0.35, C["purple_l"], C["purple"], "🧠 EEG\nEncoder", "EEGNet · 10→128"),
        (0.05, 0.1, 0.22, 0.35, C["blue_l"],   C["blue"],   "💪 Periferik\nEncoder", "1D-CNN · 6→128"),
        (0.05, -0.4, 0.22, 0.35, C["amber_l"],  C["amber"],  "📊 Behavioral\nEncoder", "MLP · 9→32"),
        (0.55, 0.1, 0.28, 0.65, C["teal_l"],   C["teal"],   "⚡ Contrastive\nFusion", "InfoNCE+Attention"),
        (0.88, 0.1, 0.1,  0.35, "#F0F9F0",     C["teal"],   "✓ Güven\nSınıfı", "3 sınıf"),
    ]
    for x, y, w, h, bg, border, label, sub in boxes:
        arch_fig.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h,
            fillcolor=bg, line=dict(color=border, width=2))
        arch_fig.add_annotation(x=x+w/2, y=y+h/2+0.05,
            text=f"<b>{label}</b>", showarrow=False,
            font=dict(size=12, color=border), align="center")
        arch_fig.add_annotation(x=x+w/2, y=y+h/2-0.1,
            text=sub, showarrow=False,
            font=dict(size=10, color="#64748B"), align="center")

    # Oklar
    for x0,y0,x1,y1 in [(0.27,0.775,0.55,0.5),(0.27,0.275,0.55,0.35),(0.27,-0.225,0.55,0.25),(0.83,0.35,0.88,0.35)]:
        arch_fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=2,
            arrowcolor="#94A3B8")

    arch_fig.update_layout(
        height=280, margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(range=[0,1], visible=False),
        yaxis=dict(range=[-0.5,1.1], visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(arch_fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Adımlar
    st.markdown('<div class="sec">Nasıl çalışıyor?</div>', unsafe_allow_html=True)
    for color, num, title, desc in [
        (C["purple"],"01","Veri Toplama",
         "CoSuBio: 34 katılımcı, 3 güven fazı. EEG + EDA + BVP + sıcaklık + ivmeölçer eş zamanlı kayıt. Oyun performans skorları ve öz-bildirim ölçekleri de toplandı."),
        (C["blue"],"02","Özellik Çıkarma",
         "EEG'den 10 frekans bandı gücü. Periferik sinyaller 128 Hz'e normalize edildi. 4 saniyelik kayan pencerelerle 63.974 epoch oluşturuldu."),
        (C["teal"],"03","InfoNCE Contrastive Fusion",
         "Aynı kişinin aynı andaki EEG + periferik pozitif çift. InfoNCE loss iki modaliteyi ortak gizil uzayda hizalar. Cross-attention 3 embedding'i birleştirir."),
        (C["amber"],"04","Değerlendirme",
         "Test setinde %70.7, F1=0.702. DEAP zero-shot: %41.5. LOSO cross-validation devam ediyor."),
    ]:
        st.markdown(f"""
        <div class="step" style="border-left:4px solid {color}">
          <div class="step-n" style="color:{color}">{num}</div>
          <div><div class="step-t">{title}</div>
          <div class="step-d">{desc}</div></div>
        </div>""", unsafe_allow_html=True)

    # Behavioral chart
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec">Davranışsal Performans — 3. Encoder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Hocamızın tavsiyesiyle eklendi · 7 oyun skoru + 2 öz-bildirim ölçeği</div>', unsafe_allow_html=True)

    cb1, cb2 = st.columns([3,2], gap="medium")
    with cb1:
        bpf = go.Figure()
        for m, vals, col in [("Dikkat",[72,85,58],C["teal"]),("Bellek",[68,79,55],C["blue"]),("Hız",[75,88,62],C["amber"])]:
            bpf.add_trace(go.Bar(name=m, x=["Nötr","Güven Artırıcı","Güven Azaltıcı"],
                y=vals, marker_color=col, opacity=0.9,
                text=[str(v) for v in vals], textposition="outside"))
        bpf.update_layout(barmode="group", height=270,
            margin=dict(l=10,r=10,t=20,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12,color="#1A2E4A"),
            legend=dict(orientation="h",y=1.12,font=dict(size=12)),
            yaxis=dict(range=[0,105],showgrid=True,gridcolor="#E2E8F0",zeroline=False),
            xaxis=dict(showgrid=False))
        st.plotly_chart(bpf, use_container_width=True, config={"displayModeBar":False})

    with cb2:
        st.markdown(f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
          padding:18px;margin-top:4px">
          <div style="font-size:13px;font-weight:700;color:#1A2E4A;margin-bottom:12px">
            9 özellik → 32-dim vektör</div>
          <div style="font-size:12px;color:#64748B;line-height:1.9">
            <b style="color:{C['teal']}">Oyun Skorları (7)</b><br>
            Dikkat · Esneklik · Bellek<br>
            Problem Çözme · Hız · Bulmaca<br><br>
            <b style="color:{C['blue']}">Öz-Bildirim (2)</b><br>
            Rosenberg Öz-Saygı<br>Genel Öz-Yeterlik
          </div>
          <div style="background:{C['teal_l']};border-radius:8px;padding:10px 12px;margin-top:12px">
            <div style="font-size:12px;font-weight:700;color:{C['teal_d']}">BehavioralEncoder</div>
            <div style="font-size:11px;color:{C['teal']};margin-top:2px">Linear·LayerNorm·ELU · 9→64→32</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Özet sonuç
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{C['teal_l']};border:1.5px solid {C['teal']};
      border-radius:14px;padding:22px 28px;text-align:center">
      <div style="font-size:13px;font-weight:700;color:{C['teal_d']};
        letter-spacing:1px;margin-bottom:10px">SONUÇ</div>
      <div style="font-size:14px;color:{C['teal_d']};line-height:1.9">
        <b>34 subject · 3 güven fazı · 63.974 epoch</b><br>
        EEG + periferik + davranışsal performans birlikte analiz edildi<br>
        <span style="font-size:22px;font-weight:800">%70.7 test doğruluğu</span><br>
        <span style="font-size:12px;color:{C['teal']}">
          Rastgele tahmin %33.3 → modelimiz +37.4 puan önde
        </span>
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# CANLI DEMO
# ════════════════════════════════════════════════════════════
elif page == "⚡ Canlı Demo":

    st.markdown("""<div style="padding:18px 0 14px;border-bottom:1px solid #E2E8F0;margin-bottom:18px">
      <div class="sec" style="margin-bottom:3px">Canlı Demo</div>
      <div class="sec-s" style="margin-bottom:0">Sentetik sinyal üret → model gerçek zamanlı güven durumunu tahmin eder</div>
    </div>""", unsafe_allow_html=True)

    # Metrikler
    m1,m2,m3,m4 = st.columns(4)
    for col,lbl,val,sub,color,bg in zip([m1,m2,m3,m4],
        ["Test Accuracy","Test Macro F1","Eğitim Verisi","Parametre"],
        ["70.7%","0.702","63.9K","251K"],
        ["+37.4% rastgele üzerinde","3-sınıf dengeli","epoch · 34 subject","hafif model"],
        [C["teal"],C["blue"],C["purple"],C["amber"]],
        [C["teal_l"],C["blue_l"],C["purple_l"],C["amber_l"]]):
        with col:
            st.markdown(f"""<div class="mc" style="background:{bg}">
              <div class="mc-lbl" style="color:{color}">{lbl}</div>
              <div class="mc-val" style="color:{color}">{val}</div>
              <div class="mc-sub" style="color:{color}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="arc">
      <div class="arc-s" style="background:{C['purple_l']}">
        <div class="arc-sl" style="color:{C['purple_d']}">EEG Encoder</div>
        <div class="arc-ss">EEGNet · 10→128-d</div></div>
      <div class="arc-a">→</div>
      <div class="arc-s" style="background:{C['teal_l']}">
        <div class="arc-sl" style="color:{C['teal_d']}">Contrastive Fusion</div>
        <div class="arc-ss">InfoNCE + Cross-Attn</div></div>
      <div class="arc-a">←</div>
      <div class="arc-s" style="background:{C['blue_l']}">
        <div class="arc-sl" style="color:{C['blue_d']}">Periferik Encoder</div>
        <div class="arc-ss">1D-CNN · 6→128-d</div></div>
      <div class="arc-a">→</div>
      <div class="arc-s" style="background:{C['teal_l']}">
        <div class="arc-sl" style="color:{C['teal_d']}">Güven Sınıfı</div>
        <div class="arc-ss">nötr / pozitif / negatif</div></div>
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_res = st.columns(2, gap="medium")

    with col_ctrl:
        st.markdown("**Sinyal Ayarları**")
        mode = st.radio("", ["Sentetik sinyal üret",".npy dosyası yükle"],
                        horizontal=True, label_visibility="collapsed")
        if mode == "Sentetik sinyal üret":
            level = st.select_slider("Güven durumu",
                options=["Düşük","Nötr","Yüksek"], value="Yüksek")
            noise = st.slider("Gürültü", 0.1, 2.5, 0.6, 0.05)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("⚡ Üret & Tahmin Et", type="primary", use_container_width=True):
                    if model_ok:
                        eeg, per = make_sig(level, noise)
                        res = run_inf(mdl, eeg, per)
                        st.session_state.update({"eeg":eeg,"periph":per,"result":res})
                    else:
                        st.error("Model yüklenemedi")
            with b2:
                if st.button("Temizle", use_container_width=True):
                    for k in ["eeg","periph","result"]: st.session_state.pop(k,None)
                    st.rerun()
        else:
            ef = st.file_uploader("EEG .npy (10,512)",type=["npy"])
            pf = st.file_uploader("Periferik .npy (6,512)",type=["npy"])
            if ef and pf and st.button("Tahmin Et",type="primary",use_container_width=True):
                eeg = np.load(ef).astype(np.float32)
                per = np.load(pf).astype(np.float32)
                res = run_inf(mdl,eeg,per)
                st.session_state.update({"eeg":eeg,"periph":per,"result":res})

        if "eeg" in st.session_state:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            te, tp = st.tabs(["EEG Kanalları","Periferik"])
            bands = ["Delta","Theta","Alpha1","Alpha2","Beta1","Beta2","Gamma1","Gamma2","Attention","Meditation"]
            perms = ["Acc-X","Acc-Y","Acc-Z","BVP","EDA","Temperature"]
            ecols = [C["purple"],C["blue"],C["teal"],"#639922",C["amber"],C["coral"],"#D4537E","#7F77DD","#5F5E5A","#444441"]
            pcols = [C["purple"],C["blue"],C["teal"],C["coral"],"#639922",C["amber"]]
            with te:
                for i,(n,c) in enumerate(zip(bands,ecols)):
                    pw = float(np.mean(st.session_state["eeg"][i]**2))
                    st.markdown(f"""<div class="chip" style="background:{c}12;border-left:3px solid {c}">
                      <div class="chip-n" style="color:{c}">{n}</div>
                      <div class="chip-b" style="color:{c}99">güç: {pw:.4f}</div>
                    </div>""", unsafe_allow_html=True)
            with tp:
                for i,(n,c) in enumerate(zip(perms,pcols)):
                    rms = float(np.sqrt(np.mean(st.session_state["periph"][i]**2)))
                    st.markdown(f"""<div class="chip" style="background:{c}12;border-left:3px solid {c}">
                      <div class="chip-n" style="color:{c}">{n}</div>
                      <div class="chip-b" style="color:{c}99">RMS: {rms:.4f}</div>
                    </div>""", unsafe_allow_html=True)

    with col_res:
        st.markdown("**Tahmin Sonucu**")
        if "result" not in st.session_state:
            st.markdown("""
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
              padding:56px 24px;text-align:center">
              <div style="font-size:42px;margin-bottom:12px">🧠</div>
              <div style="font-size:15px;font-weight:700;color:#1A2E4A">Henüz sinyal yok</div>
              <div style="font-size:13px;color:#64748B;margin-top:6px">
                Sol panelden sinyal üretin</div>
            </div>""", unsafe_allow_html=True)
        else:
            res = st.session_state["result"]
            cls = {0:(C["red"],"#FCEBEB",C["red_d"],"↓","Düşük Güven"),
                   1:(C["gray"],C["gray_l"],"#444441","→","Nötr"),
                   2:(C["teal"],C["teal_l"],C["teal_d"],"↑","Yüksek Güven")}
            col,bg,tc,icon,label = cls[res["pred"]]
            st.markdown(f"""
            <div class="pred-box" style="background:{bg};border:1.5px solid {col}">
              <div style="font-size:38px">{icon}</div>
              <div>
                <div style="font-size:12px;color:{tc};font-weight:600">Tahmin</div>
                <div class="pred-cls" style="color:{tc}">{label}</div>
                <div style="font-size:12px;color:{tc}88;margin-top:2px">
                  Kesinlik: {res['probs'][res['pred']]*100:.1f}%</div>
              </div>
            </div>""", unsafe_allow_html=True)

            for lbl,pct,c in [("Düşük Güven",res["probs"][0]*100,C["red"]),
                                ("Nötr",       res["probs"][1]*100,C["gray"]),
                                ("Yüksek Güven",res["probs"][2]*100,C["teal"])]:
                st.markdown(f"""<div class="prob-h"><span>{lbl}</span><span>{pct:.1f}%</span></div>
                <div class="prob-tr"><div class="prob-f" style="width:{pct:.1f}%;background:{c}"></div></div>
                """, unsafe_allow_html=True)

            st.markdown(f"""<div class="egrid">
              <div class="ebox"><div class="elbl">EEG norm</div>
                <div class="eval">{res['eeg_norm']:.2f}</div></div>
              <div class="ebox"><div class="elbl">Periferik norm</div>
                <div class="eval">{res['periph_norm']:.2f}</div></div>
              <div class="ebox"><div class="elbl">Cosine benzerlik</div>
                <div class="eval">{res['cos_sim']:.3f}</div></div>
              <div class="ebox"><div class="elbl">Inference süresi</div>
                <div class="eval">{res['ms']:.1f} ms</div></div>
            </div>""", unsafe_allow_html=True)

    if "eeg" in st.session_state:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        t_ = np.linspace(0,4,512)
        bands_ = ["Delta","Theta","Alpha1","Alpha2","Beta1","Beta2","Gamma1","Gamma2","Attention","Meditation"]
        perms_ = ["Acc-X","Acc-Y","Acc-Z","BVP","EDA","Temperature"]
        ecols_ = [C["purple"],C["blue"],C["teal"],"#639922",C["amber"],C["coral"],"#D4537E","#7F77DD","#5F5E5A","#444441"]
        pcols_ = [C["purple"],C["blue"],C["teal"],C["coral"],"#639922",C["amber"]]
        sf = make_subplots(rows=2,cols=1,subplot_titles=["EEG (10 kanal)","Periferik (6 kanal)"],
            vertical_spacing=0.12, shared_xaxes=True)
        for i,(n,c) in enumerate(zip(bands_,ecols_)):
            sf.add_trace(go.Scatter(x=t_, y=st.session_state["eeg"][i]+i*2.5,
                name=n, line=dict(color=c,width=1.2),
                hovertemplate=f"<b>{n}</b><br>t=%{{x:.2f}}s<extra></extra>"), row=1,col=1)
        for i,(n,c) in enumerate(zip(perms_,pcols_)):
            sf.add_trace(go.Scatter(x=t_, y=st.session_state["periph"][i]+i*2.5,
                name=n, line=dict(color=c,width=1.4),
                hovertemplate=f"<b>{n}</b><br>t=%{{x:.2f}}s<extra></extra>"), row=2,col=1)
        sf.update_layout(height=400, margin=dict(l=10,r=10,t=40,b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11,color="#1A2E4A"),
            legend=dict(orientation="h",y=1.05,font=dict(size=10)),
            hovermode="x unified")
        sf.update_xaxes(showgrid=True,gridcolor="#E2E8F0",zeroline=False,
            title_text="Zaman (s)",row=2,col=1)
        sf.update_yaxes(showgrid=False,showticklabels=False)
        st.plotly_chart(sf, use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════
elif page == "🔬 Model":

    st.markdown("""<div style="padding:18px 0 14px;border-bottom:1px solid #E2E8F0;margin-bottom:18px">
      <div class="sec" style="margin-bottom:3px">Model Analizi</div>
      <div class="sec-s" style="margin-bottom:0">Ablasyon · mimari · cross-dataset genellenebilirlik</div>
    </div>""", unsafe_allow_html=True)

    # Ablasyon — dramatik
    st.markdown('<div class="sec">Ablasyon Çalışması</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Her bileşeni tek tek kaldırıp etkisini ölçtük — sonuçlar konuşuyor</div>', unsafe_allow_html=True)

    col_ac, col_ag = st.columns([2,3], gap="medium")
    with col_ac:
        for name,v in ABLATION.items():
            is_best = name=="Full Model (ours)"
            bw = int(v["f1"]/0.75*100)
            st.markdown(f"""
            <div class="abl {"best" if is_best else ""}">
              <div class="abl-n {"best" if is_best else ""}">
                {"✓ " if is_best else ""}{name}</div>
              <div class="abl-d {"best" if is_best else ""}">{v["desc"]}</div>
              <div class="bar-bg">
                <div style="width:{bw}%;height:100%;background:{v['color']};border-radius:5px"></div>
              </div>
              <div class="abl-sc">
                <span>Acc {v['acc']*100:.1f}%</span>
                <span style="font-weight:600;color:{'#085041' if is_best else '#64748B'}">
                  F1 {v['f1']:.3f}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_ag:
        # Dramatik bar chart
        names_a = list(ABLATION.keys())
        f1s_a   = [v["f1"] for v in ABLATION.values()]
        cols_a  = [v["color"] for v in ABLATION.values()]
        af = go.Figure()
        af.add_trace(go.Bar(x=names_a, y=f1s_a, marker_color=cols_a,
            marker_line_color="white", marker_line_width=2,
            text=[f"F1: {v:.3f}" for v in f1s_a], textposition="outside",
            hovertemplate="%{x}<br>F1: %{y:.3f}<extra></extra>"))
        af.add_hline(y=0.333, line_dash="dot", line_color=C["gray"],
            annotation_text="Rastgele (0.333)", annotation_position="top right",
            annotation_font_color=C["gray"])
        af.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=dict(size=12,color="#1A2E4A"),
            yaxis=dict(showgrid=True,gridcolor="#E2E8F0",zeroline=False,
                range=[0,0.86], title="Macro F1"),
            xaxis=dict(showgrid=False, tickangle=-15),
            title=dict(text="Ablasyon — Macro F1 Karşılaştırması",
                font=dict(size=14,color="#1A2E4A")))
        st.plotly_chart(af, use_container_width=True, config={"displayModeBar":False})

        for col_,txt in [
            (C["red"],   "EEG tek başına F1=0.206 — NeuroSky tüketici cihazı kısıtlaması"),
            (C["blue"],  "Periferik eklince +0.368 — EDA/BVP güvenle korelasyonlu"),
            (C["amber"], "Fusion +0.077 — multimodal yaklaşım doğrulandı"),
            (C["teal"],  "InfoNCE +0.051 — cross-modal hizalama projenin özgün katkısı ✓"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:7px">
              <div style="width:3px;min-height:32px;background:{col_};border-radius:2px;flex-shrink:0;margin-top:2px"></div>
              <div style="font-size:12px;color:#64748B;line-height:1.6">{txt}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # DEAP cross-dataset
    st.markdown('<div class="sec">Cross-Dataset Genellenebilirlik</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">CoSuBio\'da eğitilen modeli DEAP\'e hiç eğitim yapmadan uyguladık</div>', unsafe_allow_html=True)

    cd1,cd2,cd3 = st.columns(3)
    for col,lbl,val,sub,color,bg in zip([cd1,cd2,cd3],
        ["CoSuBio (Eğitim)","DEAP (Zero-shot)","Rastgele Baseline"],
        ["70.7%","41.5%","33.3%"],
        ["Test accuracy · F1:0.702","Zero-shot · +8.2%","3 sınıf beklenti değeri"],
        [C["teal"],C["blue"],C["gray"]],[C["teal_l"],C["blue_l"],C["gray_l"]]):
        with col:
            st.markdown(f"""<div class="mc" style="background:{bg};text-align:center">
              <div class="mc-lbl" style="color:{color}">{lbl}</div>
              <div class="mc-val" style="color:{color}">{val}</div>
              <div class="mc-sub" style="color:{color}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="background:{C['blue_l']};border:1px solid {C['blue']};
      border-radius:10px;padding:14px 16px;margin-top:6px">
      <div style="font-size:12px;color:{C['blue_d']};line-height:1.7">
        Model CoSuBio'da eğitildi, DEAP'i hiç görmeden uygulandı. <b>+8.2 puan</b>
        rastgele üzerinde — InfoNCE ile öğrenilen cross-modal temsiller kısmen genellenebilir.
        Domain shift (farklı cihaz/görev) nedeniyle tam transfer beklenmez — literatürde bilinen kısıtlama.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # LOSO Cross-Validation
    st.markdown('<div class="sec">LOSO Cross-Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-s">Leave-One-Subject-Out — 33 fold · subject-independent değerlendirme</div>', unsafe_allow_html=True)

    # Özet metrik kartları
    lc1, lc2, lc3, lc4 = st.columns(4)
    for col, lbl, val, sub, color, bg in zip(
        [lc1, lc2, lc3, lc4],
        ["Mean Accuracy", "Mean Macro F1", "Medyan F1", "İyi Subject (F1≥0.5)"],
        ["57.1%", "0.444", "0.528", "18 / 33"],
        ["±21.6%", "±0.256", "daha sağlam gösterge", "yüzde 55"],
        [C["teal"], C["blue"], C["purple"], C["amber"]],
        [C["teal_l"], C["blue_l"], C["purple_l"], C["amber_l"]],
    ):
        with col:
            st.markdown(f"""<div class="mc" style="background:{bg};text-align:center">
              <div class="mc-lbl" style="color:{color}">{lbl}</div>
              <div class="mc-val" style="color:{color};font-size:26px">{val}</div>
              <div class="mc-sub" style="color:{color}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # LOSO subject bazında grafik
    loso_data = {
        1:0.505, 2:1.000, 3:0.149, 4:0.585, 5:0.528,
        6:0.208, 7:0.754, 8:0.574, 9:0.562, 10:0.158,
        11:0.571, 12:0.572, 13:0.545, 14:0.556, 15:1.000,
        16:1.000, 17:0.559, 18:0.569, 19:0.552, 21:0.365,
        22:0.222, 23:0.429, 24:0.565, 25:0.261, 26:0.559,
        27:0.184, 28:0.051, 29:0.180, 30:0.171, 31:0.178,
        32:0.177, 33:0.190, 34:0.182,
    }
    subj_labels = [f"S{k:02d}" for k in loso_data.keys()]
    f1_vals     = list(loso_data.values())
    bar_colors  = [
        C["teal"] if v >= 0.5 else (C["amber"] if v >= 0.25 else C["red"])
        for v in f1_vals
    ]

    lf = go.Figure()
    lf.add_trace(go.Bar(
        x=subj_labels, y=f1_vals,
        marker_color=bar_colors,
        marker_line_color="white", marker_line_width=1,
        hovertemplate="%{x}<br>F1: %{y:.3f}<extra></extra>",
    ))
    lf.add_hline(y=0.444, line_dash="dash", line_color=C["blue"],
        annotation_text=f"Ortalama (0.444)", annotation_position="top left",
        annotation_font_color=C["blue"])
    lf.add_hline(y=0.333, line_dash="dot", line_color=C["gray"],
        annotation_text="Rastgele (0.333)", annotation_position="top right",
        annotation_font_color=C["gray"])
    lf.update_layout(
        height=300, margin=dict(l=10,r=10,t=30,b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(size=11, color="#1A2E4A"),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False,
                   range=[0, 1.1], title="Macro F1"),
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        title=dict(text="Subject Bazında LOSO F1 Skorları",
                   font=dict(size=13, color="#1A2E4A")),
    )
    st.plotly_chart(lf, use_container_width=True, config={"displayModeBar": False})

    # Renk açıklaması + yorum kutusu
    lcol1, lcol2 = st.columns([1, 2], gap="medium")
    with lcol1:
        for color, label, n in [
            (C["teal"], "F1 ≥ 0.50 — İyi", "18 subject"),
            (C["amber"], "0.25 ≤ F1 < 0.50 — Orta", "3 subject"),
            (C["red"], "F1 < 0.25 — Zayıf", "12 subject"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
              <div style="width:14px;height:14px;background:{color};border-radius:3px;flex-shrink:0"></div>
              <div style="font-size:12px;color:#1A2E4A;flex:1">{label}</div>
              <div style="font-size:11px;font-weight:600;color:{color}">{n}</div>
            </div>""", unsafe_allow_html=True)

    with lcol2:
        st.markdown(f"""
        <div style="background:{C['purple_l']};border:1px solid {C['purple']};
          border-radius:10px;padding:14px 16px">
          <div style="font-size:12px;font-weight:700;color:{C['purple_d']};margin-bottom:8px">
            Inter-subject Heterojenlik</div>
          <div style="font-size:12px;color:{C['purple_d']};line-height:1.7">
            Yüksek varyasyon (±0.256) EEG araştırmalarında bilinen bir durum:
            her bireyin sinyal örüntüsü farklı. 33 fold'un 18'inde F1 ≥ 0.5
            başarısı elde edildi. Subject 02, 15, 16'daki F1=1.0 sonuçları
            behavioral vektör dağılımından kaynaklanıyor olabilir — gelecek
            çalışmada incelenecek.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Mimari
    st.markdown('<div class="sec">Mimari Bileşenler</div>', unsafe_allow_html=True)
    mc1,mc2,mc3 = st.columns(3)
    for col,title,color,bg,items in zip([mc1,mc2,mc3],
        ["EEG Encoder — EEGNet","Contrastive Fusion","Periferik + Behavioral"],
        [C["purple"],C["teal"],C["blue"]],
        [C["purple_l"],C["teal_l"],C["blue_l"]],
        [["Temporal Conv (1×64)","Depthwise Conv (kanal×1)",
          "Separable Conv (1×16)","Projeksiyon → 128-dim","Girdi: (10, 512)"],
         ["InfoNCE Loss (τ=0.07)","Cross-Attention (4 kafa)",
          "Concat → 288-dim","MLP 288→64→3","α=0.5 loss dengesi"],
         ["1D-CNN: 32→64→128 kanal","AdaptiveAvgPool → 128-dim",
          "BehavioralEncoder: 9→32","Toplam: 288-dim birleşim",
          "Girdi: (6,512) + (9,)"]]):
        with col:
            items_html="".join(f"<li style='margin-bottom:5px'>{it}</li>" for it in items)
            st.markdown(f"""
            <div style="background:{bg};border-radius:12px;padding:16px 18px">
              <div style="font-size:13px;font-weight:700;color:{color};
                margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid {color}44">
                {title}</div>
              <ul style="font-size:12px;color:#1A2E4A;padding-left:16px;
                line-height:1.7;margin:0">{items_html}</ul>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# EĞİTİM
# ════════════════════════════════════════════════════════════
elif page == "📈 Eğitim":

    st.markdown("""<div style="padding:18px 0 14px;border-bottom:1px solid #E2E8F0;margin-bottom:18px">
      <div class="sec" style="margin-bottom:3px">Eğitim Geçmişi</div>
      <div class="sec-s" style="margin-bottom:0">50 epoch · CoSuBio · RTX 2050 GPU · W&B takibi</div>
    </div>""", unsafe_allow_html=True)

    e1,e2,e3,e4 = st.columns(4)
    for col,lbl,val,sub,color,bg in zip([e1,e2,e3,e4],
        ["En iyi Val F1","En iyi Epoch","Final Train Loss","Final Val Loss"],
        ["0.703","38","2.343","2.339"],
        ["Epoch 38","50 içinden","yakınsadı","overfit yok"],
        [C["teal"],C["purple"],C["blue"],C["amber"]],
        [C["teal_l"],C["purple_l"],C["blue_l"],C["amber_l"]]):
        with col:
            st.markdown(f"""<div class="mc" style="background:{bg}">
              <div class="mc-lbl" style="color:{color}">{lbl}</div>
              <div class="mc-val" style="color:{color}">{val}</div>
              <div class="mc-sub" style="color:{color}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Hiperparametreler
    hcols = st.columns(8)
    for col,k,v in zip(hcols,
        ["Optimizer","LR","Batch","Epoch","Scheduler","Split","Alpha (α)","Tau (τ)"],
        ["AdamW","3×10⁻⁴","64","50","Cosine","70/15/15","0.5","0.07"]):
        with col:
            st.markdown(f"""<div style="background:#FFFFFF;border:1px solid #E2E8F0;
              border-radius:8px;padding:10px;text-align:center;margin-bottom:12px">
              <div style="font-size:14px;font-weight:800;color:#1A2E4A">{v}</div>
              <div style="font-size:9px;color:#64748B;margin-top:2px">{k}</div>
            </div>""", unsafe_allow_html=True)

    epochs = list(range(1,51))
    best_ep = int(np.argmax(VAL_F1))+1
    tf = make_subplots(rows=1,cols=2,
        subplot_titles=["Eğitim & Doğrulama Kaybı","Doğrulama Macro F1"],
        horizontal_spacing=0.10)
    tf.add_trace(go.Scatter(x=epochs,y=TRAIN_LOSS,name="Train Loss",
        line=dict(color=C["blue"],width=2.5),
        hovertemplate="Epoch %{x}<br>Train: %{y:.4f}<extra></extra>"),row=1,col=1)
    tf.add_trace(go.Scatter(x=epochs,y=VAL_LOSS,name="Val Loss",
        line=dict(color=C["coral"],width=2.5,dash="dash"),
        hovertemplate="Epoch %{x}<br>Val: %{y:.4f}<extra></extra>"),row=1,col=1)
    tf.add_trace(go.Scatter(x=epochs,y=VAL_F1,name="Val F1",
        line=dict(color=C["teal"],width=3),
        fill="tozeroy",fillcolor="rgba(29,158,117,0.08)",
        hovertemplate="Epoch %{x}<br>F1: %{y:.3f}<extra></extra>"),row=1,col=2)
    tf.add_trace(go.Scatter(x=[best_ep],y=[max(VAL_F1)],mode="markers+text",
        marker=dict(color=C["red"],size=12,symbol="star"),
        text=[f"En iyi: {max(VAL_F1):.3f}"],textposition="top right",
        name=f"Best (ep {best_ep})",
        hovertemplate=f"Best epoch {best_ep}<extra></extra>"),row=1,col=2)
    tf.update_layout(height=340,margin=dict(l=10,r=10,t=50,b=30),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12,color="#1A2E4A"),
        legend=dict(orientation="h",y=-0.15,font=dict(size=11)),
        hovermode="x unified")
    tf.update_yaxes(showgrid=True,gridcolor="#E2E8F0",zeroline=False)
    tf.update_xaxes(showgrid=False,title_text="Epoch")
    st.plotly_chart(tf, use_container_width=True, config={"displayModeBar":False})

    st.markdown(f"""<div style="background:{C['teal_l']};border:1px solid {C['teal']};
      border-radius:10px;padding:16px 20px">
      <div style="font-size:13px;font-weight:700;color:{C['teal_d']};margin-bottom:8px">
        Eğrileri nasıl yorumlamalı?</div>
      <div style="font-size:12px;color:{C['teal_d']};line-height:1.9">
        <b>Train loss sürekli düşüyor</b> (2.66→2.34) — model tutarlı öğreniyor. &nbsp;
        <b>Val loss train'i takip ediyor</b> — overfitting yok. &nbsp;
        <b>Val F1 epoch 38'de zirve</b> (0.703) — en iyi checkpoint kaydedildi. &nbsp;
        <b>Test %70.7 > Train %69.4</b> — model gerçek örüntüleri öğrenmiş, ezberlemiş değil.
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# VERİ
# ════════════════════════════════════════════════════════════
elif page == "🗂 Veri":

    st.markdown("""<div style="padding:18px 0 14px;border-bottom:1px solid #E2E8F0;margin-bottom:18px">
      <div class="sec" style="margin-bottom:3px">Veri Setleri</div>
      <div class="sec-s" style="margin-bottom:0">CoSuBio (birincil) · DEAP (cross-dataset — tamamlandı ✓)</div>
    </div>""", unsafe_allow_html=True)

    va, vb = st.columns(2, gap="medium")
    with va:
        st.markdown(f"""<div style="border:1.5px solid {C['teal']};border-radius:14px;padding:20px;margin-bottom:12px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
            <div style="background:{C['teal']};color:#fff;border-radius:6px;
              padding:4px 12px;font-size:11px;font-weight:700">BİRİNCİL</div>
            <div style="font-size:18px;font-weight:800;color:#1A2E4A">CoSuBio</div>
            <div style="font-size:11px;color:{C['teal']};margin-left:auto;font-weight:600">2026 · Yeni!</div>
          </div>""", unsafe_allow_html=True)
        for lbl,val in [("Katılımcı","34 subject"),("Güven fazları","Neutral / Positive / Negative"),
            ("EEG","10 band gücü (NeuroSky · 1 Hz)"),("Periferik","EDA, BVP, Sıcaklık, ACC-X/Y/Z"),
            ("Behavioral","7 oyun skoru + 2 öz-bildirim"),("Epoch","63.974 · 4 sn · 512 sample")]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;
              padding:7px 0;border-bottom:0.5px solid #E2E8F0;font-size:13px">
              <span style="color:#64748B">{lbl}</span>
              <span style="color:#1A2E4A;font-weight:600">{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        df = go.Figure(go.Bar(
            x=["Nötr","Pozitif","Negatif"], y=[24478,18801,20695],
            marker_color=[C["gray"],C["teal"],C["red"]],
            text=["24.478","18.801","20.695"],textposition="outside"))
        df.update_layout(height=210,margin=dict(l=10,r=10,t=30,b=10),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            title="Sınıf dağılımı",title_font=dict(size=13,color="#1A2E4A"),
            font=dict(size=12,color="#1A2E4A"),showlegend=False,
            yaxis=dict(showgrid=True,gridcolor="#E2E8F0",zeroline=False),
            xaxis=dict(showgrid=False))
        st.plotly_chart(df, use_container_width=True, config={"displayModeBar":False})

    with vb:
        st.markdown(f"""<div style="border:1.5px solid {C['blue']};border-radius:14px;padding:20px;margin-bottom:12px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
            <div style="background:{C['blue']};color:#fff;border-radius:6px;
              padding:4px 12px;font-size:11px;font-weight:700">ÇAPRAZ-VERİSETİ ✓</div>
            <div style="font-size:18px;font-weight:800;color:#1A2E4A">DEAP</div>
          </div>""", unsafe_allow_html=True)
        for lbl,val,bold in [("Katılımcı","32 subject",False),
            ("Görev","40 müzik videosu izleme",False),
            ("EEG","32 kanal · 128 Hz",False),
            ("Periferik","EDA, BVP, EMG, Sıcaklık, EOG",False),
            ("Epoch","37.120 · zero-shot",False),
            ("Test Accuracy","41.5% (sıfır eğitim)",True),
            ("Test Macro F1","0.325",True),
            ("Baseline üstünde","+8.2 puan",True)]:
            c_ = C["blue"] if bold else "#64748B"
            w_ = "700" if bold else "500"
            st.markdown(f"""<div style="display:flex;justify-content:space-between;
              padding:7px 0;border-bottom:0.5px solid #E2E8F0;font-size:13px">
              <span style="color:#64748B">{lbl}</span>
              <span style="color:{c_};font-weight:{w_}">{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        cf = go.Figure(go.Bar(
            x=["CoSuBio","DEAP\nZero-shot","Rastgele"],
            y=[70.7,41.5,33.3],
            marker_color=[C["teal"],C["blue"],C["gray"]],
            text=["70.7%","41.5%","33.3%"],textposition="outside"))
        cf.add_hline(y=33.3,line_dash="dot",line_color=C["gray"])
        cf.update_layout(height=210,margin=dict(l=10,r=10,t=30,b=10),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            title="Cross-dataset transfer",title_font=dict(size=13,color="#1A2E4A"),
            font=dict(size=12,color="#1A2E4A"),showlegend=False,
            yaxis=dict(range=[0,85],showgrid=True,gridcolor="#E2E8F0",zeroline=False),
            xaxis=dict(showgrid=False))
        st.plotly_chart(cf, use_container_width=True, config={"displayModeBar":False})

        st.markdown(f"""<div style="background:{C['blue_l']};border-radius:10px;padding:14px 16px">
          <div style="font-size:12px;color:{C['blue_d']};line-height:1.7">
            Model CoSuBio'da eğitildi, DEAP'i hiç görmeden uygulandı.
            <b>+8.2 puan</b> rastgele üzerinde. Domain shift nedeniyle tam transfer
            beklenmez — bu literatürde bilinen bir kısıtlamadır.
          </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HAKKINDA
# ════════════════════════════════════════════════════════════
elif page == "ℹ Hakkında":

    st.markdown("""<div style="padding:18px 0 14px;border-bottom:1px solid #E2E8F0;margin-bottom:18px">
      <div class="sec" style="margin-bottom:3px">Proje Hakkında</div>
      <div class="sec-s" style="margin-bottom:0">CSE0540 Derin Öğrenme · Mayıs 2026 · Damla Gözde Genç</div>
    </div>""", unsafe_allow_html=True)

    hl, hr = st.columns([3,2], gap="medium")
    with hl:
        st.markdown(f"""<div style="background:{C['purple_l']};border-radius:12px;padding:20px;margin-bottom:16px">
          <div style="font-size:16px;font-weight:800;color:{C['purple_d']};margin-bottom:10px">
            Confidence-State Representation Learning</div>
          <div style="font-size:13px;color:{C['purple_d']};line-height:1.75">
            EEG band güçleri, periferik biyosinyaller ve davranışsal performans verilerini
            multimodal derin öğrenme ile birleştirerek güven durumu sınıflandırması yaptık.
            Özgün katkı: <b>InfoNCE contrastive loss</b> ile iki modaliteyi ortak gizil uzayda hizalamak.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Referanslar**")
        for name,authors,journal,color in [
            ("EEGNet","Lawhern et al. (2018)","J. Neural Engineering",C["purple"]),
            ("SimCLR / InfoNCE","Chen et al. (2020)","ICML 2020",C["blue"]),
            ("DEAP Dataset","Koelstra et al. (2012)","IEEE T-AFFC",C["teal"]),
            ("CoSuBio Dataset","2026","Data in Brief",C["amber"]),
            ("Multimodal Review","Pillalamarri et al. (2025)","AI Review",C["purple"]),
        ]:
            st.markdown(f"""<div style="display:flex;gap:12px;padding:8px 0;
              border-bottom:0.5px solid #E2E8F0;align-items:flex-start">
              <div style="background:{color}18;color:{color};border-radius:6px;
                padding:3px 10px;font-size:10px;font-weight:700;flex-shrink:0;margin-top:1px">{name}</div>
              <div>
                <div style="font-size:13px;font-weight:600;color:#1A2E4A">{authors}</div>
                <div style="font-size:11px;color:#64748B">{journal}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with hr:
        st.markdown(f"""<div style="border:1.5px solid {C['teal']};border-radius:12px;padding:18px;margin-bottom:14px">
          <div style="font-size:13px;font-weight:700;color:{C['teal_d']};margin-bottom:12px">Özet Sonuçlar</div>""",
          unsafe_allow_html=True)
        for lbl,val,delta,color,bg in [
            ("Test Accuracy","70.7%","+37.4% vs random",C["teal"],C["teal_l"]),
            ("Test Macro F1","0.702","+0.496 vs EEG-only",C["blue"],C["blue_l"]),
            ("DEAP Zero-shot","41.5%","+8.2% vs random",C["purple"],C["purple_l"]),
            ("Parametre","251K","hafif · etkili",C["amber"],C["amber_l"]),
        ]:
            st.markdown(f"""<div class="mc" style="background:{bg};margin-bottom:8px">
              <div class="mc-lbl" style="color:{color}">{lbl}</div>
              <div class="mc-val" style="color:{color};font-size:24px">{val}</div>
              <div class="mc-sub" style="color:{color}">{delta}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:14px 16px">
          <div style="font-size:12px;font-weight:700;color:#1A2E4A;margin-bottom:10px">Tech Stack</div>
          <div style="display:flex;flex-wrap:wrap;gap:6px">
            {"".join(f'<span style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:20px;padding:3px 10px;font-size:11px;color:#64748B">{t}</span>' for t in ["PyTorch","EEGNet","1D-CNN","InfoNCE","Streamlit","Plotly","W&B","scikit-learn","CoSuBio","DEAP"])}
          </div>
        </div>""", unsafe_allow_html=True)
