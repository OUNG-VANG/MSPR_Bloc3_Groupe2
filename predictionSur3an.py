import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Lecture des données à partir d'un fichier CSV
df = pd.read_csv('Data/resultat_jointure.csv')

# 2. Préparation des données
# Encodage des variables catégorielles
label_encoder = LabelEncoder()
df['parti_politique_vainq2017'] = label_encoder.fit_transform(df['parti_politique_vainq2017'])
df['parti_politique_vainq2022'] = label_encoder.transform(df['parti_politique_vainq2022'])

# Sélectionner les fonctionnalités pertinentes
features = df[['2017', '2018', '2019', '2020', '2021', '2022',
               'Hommes_0-19', 'Hommes_20-39', 'Hommes_40-59',
               'Hommes_60-74', 'Hommes_75plus', 'Femmes_0-19',
               'Femmes_20-39', 'Femmes_40-59', 'Femmes_60-74',
               'Femmes_75plus', 'Population_Total',
               'emploi_2007', 'emploi_2008', 'emploi_2009',
               'emploi_2010', 'emploi_2011', 'emploi_2012',
               'emploi_2013', 'emploi_2014', 'emploi_2015',
               'emploi_2016', 'emploi_2017', 'emploi_2018',
               'emploi_2019', 'emploi_2020', 'emploi_2021',
               'emploi_2022']]

# Combiner les données de 2017 et 2022 pour créer une seule cible
df_2017 = features.copy()
df_2017['target'] = df['parti_politique_vainq2017']
df_2017['year'] = 2017

df_2022 = features.copy()
df_2022['target'] = df['parti_politique_vainq2022']
df_2022['year'] = 2022

# Combiner les deux DataFrames
combined_df = pd.concat([df_2017, df_2022], ignore_index=True)

# Définir les nouvelles caractéristiques (y compris l'année) et la cible
combined_features = combined_df.drop('target', axis=1)
combined_target = combined_df['target']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(combined_features, combined_target, test_size=0.2, random_state=42)

# 3. Construction et entraînement du modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 4. Prédiction des probabilités pour 2024, 2025 et 2026
def make_predictions_for_year(year, adjustment_factor):
    features_year = df[features.columns].copy()  # Copier les colonnes de features pour garantir qu'elles correspondent
    features_year['year'] = year  # Ajouter la colonne année

    # Simuler des changements dans les taux de chômage (par exemple, ajustement de +/- adjustment_factor)
    for col in ['2017', '2018', '2019', '2020', '2021', '2022']:
        features_year[col] = features_year[col] * (1 + adjustment_factor)

    predictions_proba = model.predict_proba(features_year)

    # Obtenir les noms des partis politiques prédits
    predicted_party_indices = model.predict(features_year)
    predicted_parties = label_encoder.inverse_transform(predicted_party_indices)

    # Préparer un DataFrame avec les résultats
    results = df[['Libelle_departement']].copy()
    results[f'predicted_party{year}'] = predicted_parties

    # Créer un DataFrame pour les probabilités avec les noms des partis comme colonnes
    classes = label_encoder.classes_
    proba_df = pd.DataFrame(predictions_proba, columns=classes)

    # Combiner les résultats et les probabilités
    results = pd.concat([results, proba_df], axis=1)

    # Enregistrement des résultats dans un fichier CSV
    results.to_csv(f'Data/proba_{year}.csv', index=False)

    print(f"Les résultats des prédictions pour {year} ont été enregistrés dans 'proba_{year}.csv'.")


# Faire des prédictions pour 2024, 2025, et 2026
make_predictions_for_year(2024, adjustment_factor=0.00)  # Aucun ajustement pour 2024
make_predictions_for_year(2025, adjustment_factor=0.01)  # Ajustement de 1% pour simuler les changements en 2025
make_predictions_for_year(2026, adjustment_factor=0.02)  # Ajustement de 2% pour simuler les changements en 2026
make_predictions_for_year(2027, adjustment_factor=0.03)  # Ajustement de 2% pour simuler les changements en 2027