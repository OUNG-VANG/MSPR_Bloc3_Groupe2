# MSPR_Bloc3_Groupe2
 Ce code charge des données depuis un fichier CSV, sélectionne des colonnes spécifiques comme caractéristiques
 pour entraîner un modèle de classification Random Forest. La cible du modèle est le parti politique vainqueur en 2017.
 Le modèle est entraîné avec les données sélectionnées, puis il prédit les probabilités d'appartenance à chaque classe
 pour chaque exemple. Les probabilités prédites sont ensuite stockées dans un nouveau DataFrame, auquel on ajoute
 la colonne des départements, et ce DataFrame est exporté dans un fichier CSV nommé 'probabilities.csv'.



