# Dashboard_Scoring_via_API_Prediction

Modélisation et utilisation ML Flow

Conception d'un dashboard (via Streamlit) permettant de visualiser les résultats de la classification binaire (via API créé par FastAPI). Les résultats permettent de déterminer si les clients riquent ou non d'avoir des défauts de remboursement

## Modélisation
Modélisation du modèle de prédiction, les étapes sont décrites dans le notebook

## L'API 
l'API pour la prédiction est sur la branch api, elle prend l'id client puis renvoie soit la prédiction, le probabilité, le score, les features importtances locales et globale selon la requete. (déployé sur Heroku)

## Le Dashboard
le dashboard concu avec streamlit permet de visisualiser la prédiction, le score, et d'autres variables du dataset ainsi que les features importance de chaque client
![dash score](https://github.com/BlackStylz/Dashboard_Scoring_via_API_Prediction/assets/92699143/fb3f201c-2986-42a7-95b2-cb0eb55c7399)
