# MSPR_Bloc3_Groupe2
 le script 'RandomForest.py' charge des données depuis le fichier 'resultat_jointure.csv', sélectionne des colonnes spécifiques comme caractéristiques pour entraîner un modèle de classification Random 
 Forest. 
 La cible du modèle est le parti politique vainqueur en 2017.
 Le modèle est entraîné avec les données sélectionnées, puis il prédit les probabilités d'appartenance à chaque classe pour chaque exemple. Les probabilités prédites sont ensuite stockées dans un nouveau 
 DataFrame, auquel on ajoute la colonne des départements, les partis politique, et ce DataFrame est exporté dans un fichier CSV nommé 'probabilities.csv'.

 le script 'predictionSun3an.py' lit des données à partir d'un fichier CSV, encode les variables catégorielles, et sélectionne les caractéristiques pertinentes pour entraîner un modèle de classification 
 Random Forest. Il combine les données de 2017 et 2022 pour créer une seule cible, puis divise les données en ensembles d'entraînement et de test.
 Après avoir entraîné le modèle, le script prédit les probabilités des résultats politiques pour les années 2024, 2025, 2026, et 2027, en simulant des ajustements dans les taux de chômage pour ces années. 
 Les résultats des  prédictions, y compris les probabilités des différents partis politiques, sont sauvegardés dans des fichiers CSV.




