# Dashboard_Scoring_via_API_Prediction
Conception d'un dashboard (via Streamlit) permettant de visualiser les résultats de la classification binaire (via API créé par FastAPI). Les résultats permettent de déterminer si les clients riquent ou non d'avoir des défauts de remboursement


# L'API 
l'API pour la prédiction est sur la branch api, elle prend l'id client puis renvoie soit la prédiction, le probabilité, le score, les features importtances locales et globale selon la requete. (déployé sur Heroku)

# Le Dashboard
le dashboard concu avec streamlit permet de visisualiser la prédiction, le score, et d'autres variables du dataset ainsi que les features importance de chaque client
