import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriScan · Analyse Nutritionnelle",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Typography ─────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown {
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Hero header ────────────────────────────────────── */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .hero-container h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(110deg, #667eea, #764ba2 40%, #f093fb 70%, #667eea);
        background-size: 220% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: hero-shine 6s linear infinite;
    }
    @keyframes hero-shine { to { background-position: 220% center; } }
    .hero-badge {
        display: inline-block;
        font-size: 0.72rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #b794f4;
        border: 1px solid rgba(183, 148, 244, 0.35);
        background: rgba(183, 148, 244, 0.07);
        border-radius: 999px;
        padding: 0.3rem 0.95rem;
        margin-bottom: 0.9rem;
    }
    .hero-subtitle {
        color: #8892b0 !important;
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* ── Glass card ──────────────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
    }
    .glass-card h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Section divider ────────────────────────────────── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
        margin: 1.5rem 0;
    }

    /* ── Streamlit widget overrides ──────────────────────── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #e0e0e0 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }

    /* ── Primary button ─────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        width: 100%;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }

    /* ── Result card ─────────────────────────────────────── */
    .result-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(16px);
        text-align: center;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        line-height: 1.3;
    }
    .result-sublabel {
        color: #8892b0 !important;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* ── BMI gauge ───────────────────────────────────────── */
    .bmi-container {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    .bmi-value {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
    }
    .bmi-label {
        text-align: center;
        color: #8892b0 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }
    .bmi-bar-bg {
        height: 8px;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .bmi-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }

    /* ── Probability bars ────────────────────────────────── */
    .prob-row {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        gap: 0.8rem;
    }
    .prob-label {
        min-width: 140px;
        font-size: 0.82rem;
        color: #b0b0c0 !important;
        text-align: right;
    }
    .prob-bar-bg {
        flex: 1;
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s ease;
    }
    .prob-value {
        min-width: 45px;
        font-size: 0.82rem;
        color: #8892b0 !important;
        font-weight: 500;
    }

    /* ── Advice card ──────────────────────────────────────── */
    .advice-card {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin: 1rem 0;
    }
    .advice-card p {
        margin: 0.3rem 0 !important;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* ── Metric badge ────────────────────────────────────── */
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ── Accuracy banner ─────────────────────────────────── */
    .accuracy-banner {
        text-align: center;
        padding: 0.6rem;
        margin-bottom: 1rem;
        border-radius: 10px;
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .accuracy-banner span {
        color: #667eea !important;
        font-weight: 700;
    }

    /* ── Footer ──────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #4a4a6a !important;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), "ObesityDataSet_raw_and_data_sinthetic.csv")
    df = pd.read_csv(csv_path)

    # Encoders
    encodeurs = {}
    cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
                'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encodeurs[col] = le

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, encodeurs, list(X.columns), accuracy


model, encodeurs, feature_cols, accuracy = load_and_train()

# ─────────────────────────────────────────────────────────────
# CLASS LABELS & COLORS
# ─────────────────────────────────────────────────────────────
CLASS_INFO = {
    'Insufficient_Weight': {
        'fr': 'Poids insuffisant',
        'color': '#64b5f6',
        'gradient': 'linear-gradient(135deg, #4fc3f7, #0288d1)',
        'emoji': '🔵',
        'advice': "Votre profil indique un poids en dessous de la normale. Pensez à consulter un nutritionniste pour adapter votre alimentation et atteindre un poids santé."
    },
    'Normal_Weight': {
        'fr': 'Poids normal',
        'color': '#81c784',
        'gradient': 'linear-gradient(135deg, #66bb6a, #2e7d32)',
        'emoji': '🟢',
        'advice': "Félicitations ! Votre profil correspond à un poids dans la norme. Continuez à maintenir vos bonnes habitudes alimentaires et votre activité physique."
    },
    'Overweight_Level_I': {
        'fr': 'Surpoids niveau I',
        'color': '#fff176',
        'gradient': 'linear-gradient(135deg, #ffee58, #f9a825)',
        'emoji': '🟡',
        'advice': "Votre profil montre un léger surpoids. Une alimentation plus équilibrée et un peu plus d'activité physique régulière pourraient vous aider."
    },
    'Overweight_Level_II': {
        'fr': 'Surpoids niveau II',
        'color': '#ffb74d',
        'gradient': 'linear-gradient(135deg, #ffa726, #ef6c00)',
        'emoji': '🟠',
        'advice': "Votre profil indique un surpoids modéré. Il est recommandé de consulter un professionnel de santé pour un suivi personnalisé."
    },
    'Obesity_Type_I': {
        'fr': 'Obésité type I',
        'color': '#ff8a65',
        'gradient': 'linear-gradient(135deg, #ff7043, #d84315)',
        'emoji': '🔶',
        'advice': "Votre profil correspond à une obésité de type I. Un accompagnement médical et nutritionnel est fortement recommandé pour améliorer votre santé."
    },
    'Obesity_Type_II': {
        'fr': 'Obésité type II',
        'color': '#ef5350',
        'gradient': 'linear-gradient(135deg, #e53935, #b71c1c)',
        'emoji': '🔴',
        'advice': "Votre profil indique une obésité de type II. Il est essentiel de consulter un médecin pour un suivi adapté incluant un plan alimentaire et sportif."
    },
    'Obesity_Type_III': {
        'fr': 'Obésité type III',
        'color': '#ce93d8',
        'gradient': 'linear-gradient(135deg, #ab47bc, #6a1b9a)',
        'emoji': '🟣',
        'advice': "Votre profil correspond à une obésité sévère. Une prise en charge médicale complète est indispensable. Ne tardez pas à consulter un spécialiste."
    },
}

# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <span class="hero-badge">🔬 Machine Learning · Santé</span>
    <h1>NutriScan</h1>
    <p class="hero-subtitle">Analyse prédictive de votre profil nutritionnel basée sur vos habitudes de vie</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="accuracy-banner">
    <span>Modèle entraîné</span> · Précision de <span>{accuracy:.1%}</span> sur les données de test
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ── Profil physique ──
    st.markdown("""
    <div class="glass-card">
        <h3>👤 Profil physique</h3>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        genre = st.selectbox("Genre", ["Femme", "Homme"], index=0)
    with c2:
        age = st.number_input("Âge", min_value=10, max_value=90, value=25, step=1)

    c3, c4 = st.columns(2)
    with c3:
        taille = st.number_input("Taille (m)", min_value=1.30, max_value=2.20, value=1.70, step=0.01, format="%.2f")
    with c4:
        poids = st.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

    family_history = st.selectbox(
        "Antécédents familiaux de surpoids ?",
        ["Oui", "Non"],
        index=0
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Habitudes alimentaires ──
    st.markdown("""
    <div class="glass-card">
        <h3>🍽️ Habitudes alimentaires</h3>
    </div>
    """, unsafe_allow_html=True)

    favc = st.selectbox("Consommez-vous souvent des aliments riches en calories ?", ["Non", "Oui"], index=0)

    fcvc = st.slider("Fréquence de consommation de légumes (1 = jamais, 3 = toujours)", 1.0, 3.0, 2.0, 0.1)

    ncp = st.slider("Nombre de repas principaux par jour", 1.0, 4.0, 3.0, 0.1)

    caec = st.selectbox(
        "Grignotage entre les repas",
        ["Jamais", "Parfois", "Souvent", "Toujours"],
        index=1
    )

    ch2o = st.slider("Consommation d'eau quotidienne (litres)", 1.0, 3.0, 2.0, 0.1)

    calc = st.selectbox(
        "Consommation d'alcool",
        ["Jamais", "Parfois", "Souvent", "Toujours"],
        index=1
    )

with col_right:
    # ── Mode de vie ──
    st.markdown("""
    <div class="glass-card">
        <h3>🏃 Mode de vie</h3>
    </div>
    """, unsafe_allow_html=True)

    smoke = st.selectbox("Fumez-vous ?", ["Non", "Oui"], index=0)
    scc = st.selectbox("Surveillez-vous votre apport calorique ?", ["Non", "Oui"], index=0)

    faf = st.slider("Activité physique (jours par semaine, 0-3)", 0.0, 3.0, 1.0, 0.1)

    tue = st.slider("Temps passé devant les écrans (0-2 heures/jour)", 0.0, 2.0, 1.0, 0.1)

    mtrans = st.selectbox(
        "Moyen de transport principal",
        ["Transports en commun", "Voiture", "Marche", "Vélo", "Moto"],
        index=0
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Predict button ──
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Analyser mon profil", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ENCODE & PREDICT
# ─────────────────────────────────────────────────────────────
CAEC_MAP = {"Jamais": "no", "Parfois": "Sometimes", "Souvent": "Frequently", "Toujours": "Always"}
CALC_MAP = {"Jamais": "no", "Parfois": "Sometimes", "Souvent": "Frequently", "Toujours": "Always"}
MTRANS_MAP = {
    "Transports en commun": "Public_Transportation",
    "Voiture": "Automobile",
    "Marche": "Walking",
    "Vélo": "Bike",
    "Moto": "Motorbike",
}

if predict_btn:
    # Build raw dict
    raw = {
        'Gender': 'Female' if genre == "Femme" else 'Male',
        'Age': float(age),
        'Height': taille,
        'Weight': poids,
        'family_history_with_overweight': 'yes' if family_history == "Oui" else 'no',
        'FAVC': 'yes' if favc == "Oui" else 'no',
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': CAEC_MAP[caec],
        'SMOKE': 'yes' if smoke == "Oui" else 'no',
        'CH2O': ch2o,
        'SCC': 'yes' if scc == "Oui" else 'no',
        'FAF': faf,
        'TUE': tue,
        'CALC': CALC_MAP[calc],
        'MTRANS': MTRANS_MAP[mtrans],
    }

    # Encode categoricals
    encoded = {}
    for col in feature_cols:
        if col in encodeurs:
            encoded[col] = encodeurs[col].transform([raw[col]])[0]
        else:
            encoded[col] = raw[col]

    input_df = pd.DataFrame([encoded])[feature_cols]
    pred_encoded = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    pred_label = encodeurs['NObeyesdad'].inverse_transform([pred_encoded])[0]

    info = CLASS_INFO[pred_label]
    bmi = poids / (taille ** 2)

    # ── Results ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    r1, r2 = st.columns([3, 2], gap="large")

    with r1:
        st.markdown(f"""
        <div class="result-card">
            <p style="font-size:0.85rem; color:#8892b0; text-transform:uppercase; letter-spacing:2px; margin-bottom:0.5rem;">Résultat de l'analyse</p>
            <p class="result-label" style="color:{info['color']}">{info['emoji']} {info['fr']}</p>
            <p class="result-sublabel">Classification : {pred_label}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Probabilités par catégorie</h3>
        </div>
        """, unsafe_allow_html=True)

        classes = encodeurs['NObeyesdad'].classes_
        sorted_indices = np.argsort(proba)[::-1]

        prob_html = ""
        for idx in sorted_indices:
            cls = classes[idx]
            p = proba[idx]
            ci = CLASS_INFO[cls]
            pct = p * 100
            prob_html += f"""
            <div class="prob-row">
                <span class="prob-label">{ci['emoji']} {ci['fr']}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%; background:{ci['gradient']};"></div>
                </div>
                <span class="prob-value">{pct:.1f}%</span>
            </div>
            """

        st.markdown(prob_html, unsafe_allow_html=True)

    with r2:
        # BMI gauge
        bmi_pct = min(max((bmi - 12) / (45 - 12) * 100, 0), 100)
        if bmi < 18.5:
            bmi_color = "#64b5f6"
        elif bmi < 25:
            bmi_color = "#81c784"
        elif bmi < 30:
            bmi_color = "#ffb74d"
        else:
            bmi_color = "#ef5350"

        st.markdown(f"""
        <div class="glass-card">
            <div class="bmi-container">
                <p class="bmi-label">Indice de Masse Corporelle</p>
                <p class="bmi-value" style="color:{bmi_color}">{bmi:.1f}</p>
                <div class="bmi-bar-bg">
                    <div class="bmi-bar-fill" style="width:{bmi_pct}%; background: linear-gradient(90deg, #64b5f6, #81c784, #ffb74d, #ef5350);"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:0.4rem;">
                    <span style="font-size:0.7rem; color:#64b5f6;">Maigre</span>
                    <span style="font-size:0.7rem; color:#81c784;">Normal</span>
                    <span style="font-size:0.7rem; color:#ffb74d;">Surpoids</span>
                    <span style="font-size:0.7rem; color:#ef5350;">Obèse</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Advice
        st.markdown(f"""
        <div class="advice-card">
            <p style="font-weight:600; color:#667eea !important; margin-bottom:0.5rem !important;">💡 Recommandation</p>
            <p>{info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Conseils personnalisés selon les réponses ──
        perso = []
        if smoke == "Oui":
            perso.append(("🚭", "Vous fumez : songez à vous faire aider si vous n'arrivez pas à arrêter (Tabac Info Service, 39 89 — gratuit)."))
        if calc in ("Souvent", "Toujours"):
            perso.append(("🍷", "Consommation d'alcool élevée : réduisez-la progressivement, et n'hésitez pas à demander de l'aide si c'est difficile (Alcool Info Service, 0 980 980 930)."))
        elif calc == "Parfois":
            perso.append(("🍷", "Gardez une consommation d'alcool occasionnelle et modérée."))
        if favc == "Oui":
            perso.append(("🍔", "Aliments riches en calories fréquents : limitez-les au profit de repas faits maison."))
        if fcvc < 2:
            perso.append(("🥦", "Trop peu de légumes : visez au moins une portion à chaque repas."))
        if caec in ("Souvent", "Toujours"):
            perso.append(("🍪", "Grignotage fréquent entre les repas : privilégiez des collations saines (fruits, oléagineux) ou espacez-les."))
        if ncp < 3:
            perso.append(("🍽️", "Peu de repas structurés : des repas réguliers aident à éviter les fringales."))
        if ch2o < 1.5:
            perso.append(("💧", "Hydratation faible : visez environ 1,5 à 2 L d'eau par jour."))
        if faf < 1:
            perso.append(("🏃", "Activité physique insuffisante : visez au moins 30 min de marche ou d'exercice la plupart des jours."))
        if tue > 1:
            perso.append(("📱", "Temps d'écran élevé : pensez à des pauses actives régulières."))
        if scc == "Non" and bmi >= 25:
            perso.append(("📊", "Suivre vos apports caloriques peut vous aider à mieux vous situer."))
        if family_history == "Oui":
            perso.append(("🧬", "Antécédents familiaux de surpoids : une hygiène de vie régulière est d'autant plus bénéfique."))

        if perso:
            items = "".join(
                f'<div style="display:flex; gap:0.6rem; align-items:flex-start; margin-bottom:0.7rem;">'
                f'<span style="font-size:1.1rem; line-height:1.4;">{e}</span>'
                f'<span style="font-size:0.9rem; line-height:1.4;">{t}</span></div>'
                for e, t in perso
            )
        else:
            items = ('<div style="display:flex; gap:0.6rem; align-items:flex-start;">'
                     '<span style="font-size:1.1rem;">✅</span>'
                     '<span style="font-size:0.9rem;">Vos habitudes de vie sont globalement saines. Continuez ainsi !</span></div>')

        st.markdown(f"""
        <div class="advice-card">
            <p style="font-weight:600; color:#667eea !important; margin-bottom:0.9rem !important;">🎯 Conseils personnalisés</p>
            {items}
        </div>
        """, unsafe_allow_html=True)

        # Quick stats
        st.markdown(f"""
        <div class="glass-card" style="padding:1.2rem;">
            <p style="font-size:0.8rem; color:#8892b0; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem;">Récapitulatif</p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem;">
                <div style="text-align:center; padding:0.6rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <p style="font-size:1.3rem; font-weight:700; margin:0; color:#667eea !important;">{poids:.0f}<span style="font-size:0.8rem; font-weight:400;"> kg</span></p>
                    <p style="font-size:0.7rem; color:#8892b0 !important; margin:0;">Poids</p>
                </div>
                <div style="text-align:center; padding:0.6rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <p style="font-size:1.3rem; font-weight:700; margin:0; color:#764ba2 !important;">{taille:.2f}<span style="font-size:0.8rem; font-weight:400;"> m</span></p>
                    <p style="font-size:0.7rem; color:#8892b0 !important; margin:0;">Taille</p>
                </div>
                <div style="text-align:center; padding:0.6rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <p style="font-size:1.3rem; font-weight:700; margin:0; color:#f093fb !important;">{age}</p>
                    <p style="font-size:0.7rem; color:#8892b0 !important; margin:0;">Âge</p>
                </div>
                <div style="text-align:center; padding:0.6rem; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <p style="font-size:1.3rem; font-weight:700; margin:0; color:{bmi_color} !important;">{bmi:.1f}</p>
                    <p style="font-size:0.7rem; color:#8892b0 !important; margin:0;">IMC</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    NutriScan · Outil d'analyse prédictive · Données UCI Obesity Dataset<br>
    <span style="font-size:0.7rem;">Les résultats sont indicatifs et ne remplacent pas un avis médical professionnel.</span>
</div>
""", unsafe_allow_html=True)
