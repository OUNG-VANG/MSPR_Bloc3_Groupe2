import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Charger le DataFrame depuis le fichier CSV
df = pd.read_csv("Data/resultat_jointure.csv")

# 2. Sélectionner les fonctionnalités pertinentes
features = df[['2017', '2018', '2019', '2020', '2021', '2022', 'Hommes_0-19', 'Hommes_20-39', 'Hommes_40-59', 'Hommes_60-74', 'Hommes_75plus', 'Femmes_0-19', 'Femmes_20-39', 'Femmes_40-59', 'Femmes_60-74', 'Femmes_75plus']]

# La cible de notre modèle sera le parti politique vainqueur en 2017
target = df['parti_politique_vainq2017']

# 4. Entraîner un modèle de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(features, target)

# 5. Obtenir les probabilités pour chaque classe cible (parti politique) pour chaque département
probabilities = rf_model.predict_proba(features)

# 6. Créer un DataFrame pour stocker les probabilités
probabilities_df = pd.DataFrame(probabilities, columns=rf_model.classes_)

# Ajouter la colonne des départements au DataFrame
probabilities_df.insert(0, 'Libelle_departement', df['Libelle_departement'])

# 7. Stocker les probabilités dans un nouveau fichier CSV
probabilities_df.to_csv('probabilities.csv', index=False)

