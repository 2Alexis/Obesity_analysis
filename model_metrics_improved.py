from sklearn.metrics import r2_score, cohen_kappa_score, matthews_corrcoef, f1_score

# Entraînement
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Calcul des métriques supplémentaires
r2 = r2_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

noms_classes = encodeurs['NObeyesdad'].inverse_transform(sorted(y.unique()))

print("="*60)
print("MÉTRIQUES GLOBALES DU MODÈLE")
print("="*60)
print(f"Accuracy            : {acc:.1%}")
print(f"R² Score            : {r2:.4f}")
print(f"Cohen's Kappa       : {kappa:.4f}")
print(f"Matthews Corr. Coef : {mcc:.4f}")
print(f"F1-Score (macro)    : {f1_macro:.4f}")
print(f"F1-Score (weighted) : {f1_weighted:.4f}")
print("="*60)
print("\nRAPPORT DÉTAILLÉ PAR CLASSE :")
print(classification_report(y_test, y_pred, target_names=noms_classes))
