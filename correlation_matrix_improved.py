# Matrice de corrélation améliorée avec family_history_with_overweight
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# D'abord, encoder la variable family_history_with_overweight
# Créer une copie du dataframe pour ne pas modifier l'original
df_corr = df.copy()

# Encoder family_history_with_overweight (yes=1, no=0)
df_corr['family_history_encoded'] = df_corr['family_history_with_overweight'].map({'yes': 1, 'no': 0})

# Définir les colonnes pour la corrélation (incluant la nouvelle variable encodée)
cols_corr = ['Age', 'Height', 'Weight', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'family_history_encoded']

# Calculer la matrice de corrélation
matrice = df_corr[cols_corr].corr()

# Créer la visualisation
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(matrice, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(matrice, mask=mask, annot=True, fmt='.2f', cmap=cmap, center=0,
            square=True, linewidths=0.8, ax=ax, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.75, 'label': 'Corrélation de Pearson'})

ax.set_title('Corrélations entre variables numériques (incluant antécédents familiaux)', 
             fontweight='bold', pad=15)

# Renommer le label pour plus de clarté
labels = ax.get_yticklabels()
labels_text = [label.get_text() for label in labels]
for i, label in enumerate(labels_text):
    if label == 'family_history_encoded':
        labels_text[i] = 'Antécédents\nfamiliaux'
ax.set_yticklabels(labels_text)

# Faire de même pour les labels x
labels_x = ax.get_xticklabels()
labels_x_text = [label.get_text() for label in labels_x]
for i, label in enumerate(labels_x_text):
    if label == 'family_history_encoded':
        labels_x_text[i] = 'Antécédents\nfamiliaux'
ax.set_xticklabels(labels_x_text, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Afficher quelques corrélations intéressantes avec family_history
print("\nCorrélations avec les antécédents familiaux de surpoids :")
print("=" * 60)
family_corr = matrice['family_history_encoded'].sort_values(ascending=False)
for var, corr in family_corr.items():
    if var != 'family_history_encoded':
        print(f"{var:20s} : {corr:+.3f}")
