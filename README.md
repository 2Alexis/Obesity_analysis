# 🔬 NutriScan — Prédiction du Niveau d'Obésité

Application web de prédiction du niveau d'obésité basée sur les habitudes alimentaires et le mode de vie, utilisant un modèle **Random Forest** entraîné sur le [UCI Obesity Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).

## 📸 Aperçu

- Interface sombre avec design glassmorphism
- Formulaire interactif en français
- Résultats avec jauge IMC, probabilités par classe et conseils personnalisés
- 7 catégories prédites : du poids insuffisant à l'obésité type III

## 🚀 Installation & Lancement

### 1. Cloner le projet

```bash
git clone https://github.com/2Alexis/ObesityDataSet_raw_and_data_sinthetic.git
cd ObesityDataSet_raw_and_data_sinthetic
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
python -m streamlit run app.py
```

L'app s'ouvre automatiquement dans le navigateur à l'adresse **http://localhost:8501**.

## 📂 Structure du projet

```
├── app.py                                        # Application Streamlit
├── EDA_Obesity.ipynb                              # Notebook d'analyse exploratoire
├── ObesityDataSet_raw_and_data_sinthetic.csv      # Dataset brut (2112 individus)
├── donnees_encodees_ML.csv                        # Dataset encodé pour le ML
├── requirements.txt                               # Dépendances Python
└── README.md
```

## 🛠️ Stack technique

| Composant | Technologie |
|-----------|------------|
| Interface | Streamlit |
| Modèle ML | Random Forest (scikit-learn) |
| Données | Pandas / NumPy |
| Style | CSS custom (glassmorphism) |

## 📊 Modèle

- **Algorithme** : Random Forest Classifier (200 arbres, profondeur max 15)
- **Précision** : ~95% sur les données de test
- **Features** : 16 variables (genre, âge, taille, poids, habitudes alimentaires, activité physique, etc.)
- **Target** : 7 niveaux d'obésité (Insufficient Weight → Obesity Type III)

## ⚠️ Avertissement

Les résultats fournis par cette application sont **purement indicatifs** et ne remplacent en aucun cas un avis médical professionnel.
