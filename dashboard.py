### Script de conception du dashboard qui fait appel a une API pour la Prédiction
### Auteur: Stéphane LUBIN

# Imports
import requests
import json
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Liste et dictionnaire utile pour traitement
features_base = ['SK_ID_CURR',
                 'CODE_GENDER',
                 'NAME_EDUCATION_TYPE',
                 'NAME_FAMILY_STATUS',
                 'FLAG_OWN_REALTY',
                 'OCCUPATION_TYPE',
                 'REGION_RATING_CLIENT_W_CITY',
                 'CNT_FAM_MEMBERS',
                 'NAME_INCOME_TYPE',
                 'NAME_CONTRACT_TYPE',
                 'EXT_SOURCE_1',
                 'NAME_HOUSING_TYPE',
                 'EXT_SOURCE_2',
                 'EXT_SOURCE_3',
                 'FLAG_DOCUMENT_3']

features_rfe = ['SK_ID_CURR',
                'DAYS_BIRTH',
                'DAYS_EMPLOYED',
                'AMT_INCOME_TOTAL',
                'AMT_CREDIT',
                'AMT_ANNUITY',
                'DAYS_EMPLOYED_PERC',
                'INCOME_CREDIT_PERC',
                'INCOME_PER_PERSON',
                'ANNUITY_INCOME_PERC',
                'PAYMENT_RATE',
                'INSTAL_DPD_MAX',
                'POS_MONTHS_BALANCE_MAX']

dict_feat ={ 'Age' : ['AGE', 'Distribution Age', 'Age(Années)'],
            'Revenu Total': ['AMT_INCOME_TOTAL', 'Distribution Revenu total', 'Revenu($)'],
            'Montant crédit' : ['AMT_CREDIT', 'Distribution Crédit', 'Montant($)'],
            'Montant Assurance' : ['AMT_ANNUITY', "Distribution Assurance", 'Montant($)'],
            'Taux de paiement' : ['PAYMENT_RATE', 'Distribution taux de paiement', 'Taux de paiement(%)' ],
            'Jours de retard': ['INSTAL_DPD_MAX', 'Distribution retard Max', 'Jours'],
            'Balance Mensuelle': ['POS_MONTHS_BALANCE_MAX', 'Distribution Max', 'Balance'],
            'Score ext 1' : ['EXT_SOURCE_1', 'Distribution Score ext 1', 'Scores'],
            'Score ext 2' : ['EXT_SOURCE_2', 'Distribution Score ext 2', 'Scores'],
            'Score ext 3' : ['EXT_SOURCE_3', 'Distribution Score ext 3', 'Scores']
}

dict_cat ={ 'Situation familiale' : ['NAME_FAMILY_STATUS', 'Répartition Situation familiale', 'Status'],
        "Niveau d'études" : ['NAME_EDUCATION_TYPE', 'Répartition Niveau études', 'Niveaux'],
        'Type de crédit': ['NAME_CONTRACT_TYPE', 'Répartition type crédit', 'Types'],
        'Propriétaire de logement': ['FLAG_OWN_REALTY', 'Propriétaires ou non', 'Status'],
        'Type de Revenu': ['NAME_INCOME_TYPE', 'Répartition type de revenu', 'Types'],
        'Situation hébergement':['NAME_HOUSING_TYPE', 'Répartition type hébergement', 'Types'],
        'Emploi' : ['OCCUPATION_TYPE', 'Répartition type emploi', 'Types'],
        'Score zone géo' :['REGION_RATING_CLIENT_W_CITY', 'Répartition score géo', 'Score'],
        'Fournir le document 3' : ['FLAG_DOCUMENT_3', 'Répartition possesion document 3', 'Status']
}

@st.cache_data(persist = True)
def data_load():
## Fonction de chargement et fusion des données
    df_test= pd.read_csv('Data/test_dash.csv', index_col = 'Unnamed: 0')
    df_comp= pd.read_csv('Data/comp_dash.csv', index_col = 'Unnamed: 0')
    return df_test, df_comp
  
@st.cache_resource()
def prediction(id :int):
# Fonction de chargement du modèle via API
    r = requests.get('https://prediction-api.herokuapp.com/prediction', {'id':id})
    try:
        response = r.json()
        resultat = response
    except:
        st.write("Error from server: " + str(r.content))
        resultat = r
    return resultat

@st.cache_data(persist = True)
def shap_local(id :int):
## Fonction de features importance locales chargement du modèle via API
## Et sépare les positifs et negatifs
    res = requests.get('https://prediction-api.herokuapp.com/feat_local', {'id':id})
    try:
        response = res.json()
        resultat = response
    except:
        st.write("Error from server: " + str(res.content))
        resultat = res

    fort = []
    faible = []
    res_df = pd.DataFrame(data=resultat.items(), columns =['features', 'valeurs'])
    res_df['colors'] = ['green' if x < 0 else 'red' for x in res_df['valeurs']]
    res_df.sort_values('valeurs', inplace=True)
    res_df.reset_index(inplace=True, drop=True)
    for feat in resultat:
        if resultat[feat] < 0:
            fort.append(feat)
        elif resultat[feat] > 0:
            faible.append(feat)
        else:
            pass
    return fort, faible, res_df

@st.cache_data(persist = True)
def shap_glob():
## Fonction de features importance globales chargement du modèle via API
    res = requests.get('https://prediction-api.herokuapp.com/feat_glob')
    try:
        response = res.json()
        resultat = response
    except:
        st.write("Error from server: " + str(res.content))
        resultat = res
    resu_df = pd.DataFrame(data=resultat.items(), columns =['features', 'valeurs'])
    resu_df.sort_values('valeurs', ascending=False, inplace=True)
    resu_df.reset_index(inplace=True, drop=True)
    return resu_df


@st.cache_data(persist = True)
def shapey_display(df_g, df_l):
 ## Fonction pour afficher les features importance locales et globales
 ## Features locales 
    sns.set_theme()
    st.write("Local")
    fig, ax = plt.subplots(figsize=(20, 10), dpi= 80)
    ax = plt.gca()
    plt.hlines(y=df_l.index, xmin=0, xmax=df_l['valeurs'], alpha=0.4, color=df_l['colors'], linewidth=50)
    plt.gca().set(ylabel='$Features$', xlabel='$features Importance$')
    plt.yticks(df_l.index, df_l['features'], fontsize=18)
    st.pyplot(fig)
 
 ## Features Globales 
    st.write("Global")
    temp = shap_glob()
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_g, y=df_g['features'], x=df_g['valeurs'], color ='blue', ax=ax1)
    ax1.set(xlabel='features importance', ylabel='features')
    ax1.legend()
    st.pyplot(fig1)

@st.cache_data(persist = True)
def kde_display(train, test, comp, id):
#Fonction qui permet d'afficher les distribution des variables quantitatives
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(train[comp[0]], color ='blue', ax=ax)
    #Verification que la donnée ne soit pas manquante
    if pd.isna(test[test['SK_ID_CURR']==id][comp[0]].values):
        st.write("Comparaison impossible la donnée est manquante")
    else:
        ax.axvline(test[test['SK_ID_CURR']==id][comp[0]].values, color="green", linestyle='--')
    ax.set(title=comp[1], xlabel=comp[2], ylabel='Densité')
    ax.legend()
    st.pyplot(fig)
    
@st.cache_data(persist = True)
def bi_display(data, yval, xval, id):
    #Fonction qui permet d'afficher des scatterplot pour analyse bi-variré
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot( x= data[xval[0]], y = data[yval[0]])
    if pd.isna(data[data['SK_ID_CURR']==id][yval[0]].values) | pd.isna(data[data['SK_ID_CURR']==id][xval[0]].values):
        st.write("Comparaison impossible la donnée est manquante")
    else:
        sns.scatterplot( x= data[xval[0]].loc[data['SK_ID_CURR']==id], 
        y = data[yval[0]].loc[data['SK_ID_CURR']==id], color='r')
    ax.set(xlabel=xval[1], ylabel=xval[1])
    ax.legend()
    st.pyplot(fig)

@st.cache_data(persist = True)
def pie_bar_display(data, data_t, var, id,selec):
#Fonction qui permet d'afficher les countplot et les pie chart des variable catégorielles
## Countplot
    sns.set_theme()
    if selec == 'Countplot':
        fig, ax = plt.subplots()
        sns.set_style("darkgrid")
        ax = sns.countplot(y= data[var[0]], order=data[var[0]].value_counts().index, palette = sns.color_palette('Blues_r'))
        total = len(data[var[0]])
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2
            ax.annotate(percentage, (x, y))
        ax.set(title=var[1], ylabel=var[2], xlabel='Count')
        #ax.legend()
## Pie chart
    elif selec == 'Pie chart':
        grade = data[var[0]].value_counts()
        grade_l= grade.sort_index()
        explode = np.array([0.0 for i in range(grade_l.shape[0])])
        index_list = grade_l.index.tolist()
        # verification donnée manquante
        if pd.isna(data_t[data_t['SK_ID_CURR']==id][var[0]].values):
             st.write("Comparaison impossible la donnée est manquante")
        else:
            exp = index_list.index(data_t[data_t['SK_ID_CURR']==id][var[0]].values)
            explode[exp] = 0.1
        fig, ax = plt.subplots()
        ax.pie(grade, explode=explode, labels=grade.index, autopct='%1.1f%%', colors =sns.color_palette('Blues_r'),
                shadow=False, startangle=0)
        ax.axis('equal')
    else:
        pass

    st.pyplot(fig)

def main():
#En-tete
    st.markdown("<h1 style='text-align: center;'>DASHBOARD MODELE DE SCORING</h1>", unsafe_allow_html=True)
    st.sidebar.image('./images/Logo.png', use_column_width= 'always')
#Chargement Data
    data_test, data_comp = data_load()
#Sidebar
    rech = st.sidebar.radio('Recherche client:', ('Saisie Manuelle','Selection'), index=1)
    #Saisie Manuelle
    if rech == 'Saisie Manuelle':
        ide = st.sidebar.number_input('Numéro identifiant client',
        min_value = data_test['SK_ID_CURR'].min(),
        max_value = data_test['SK_ID_CURR'].max() )
        if ide not in data_test['SK_ID_CURR'].values:
            st.sidebar.write("Erreur de saisir le numéro client n'est pas dans la liste")
        else:
            pass
    #Liste de selection
    else:
        ide = st.sidebar.selectbox('Numéro identifiant client', data_test['SK_ID_CURR'])
        
# Listes pour pré-trier les features a afficher
    all = ['CODE_GENDER']
    base = ['AGE', 'NAME_FAMILY_STATUS','OCCUPATION_TYPE','JOB_SENIORITY','AMT_INCOME_TOTAL','AMT_ANNUITY','AMT_CREDIT']
    all.extend(base)
    details_perc = ['DAYS_EMPLOYED_PERC','INCOME_CREDIT_PERC','INCOME_PER_PERSON','ANNUITY_INCOME_PERC','INSTAL_DPD_MAX','POS_MONTHS_BALANCE_MAX','PAYMENT_RATE']
    all.extend(details_perc)
    details_inf = ['NAME_EDUCATION_TYPE','FLAG_OWN_REALTY','REGION_RATING_CLIENT_W_CITY','CNT_FAM_MEMBERS','NAME_INCOME_TYPE',
    'NAME_CONTRACT_TYPE','NAME_HOUSING_TYPE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
    all.extend(details_inf)

# Verification d'erreur et prédiction
    if ide in data_test['SK_ID_CURR'].values:
        resultat = prediction(ide)
        st.session_state['id'] = ide

##### Partie Informations a afficher
        if st.sidebar.checkbox('Afficher données client', False):
            st.markdown("<h2 style='text-align: center;'>Données du client: {}</h2>".format(ide), unsafe_allow_html=True)
            inf = ['CODE_GENDER']
            st.markdown('Groupe des informations à afficher')
            col1, col2, col3 = st.columns(3)
            #Checkbox pour afficher et enlever les données dites de base
            with col1:
                if st.checkbox('Base', False):
                    inf.extend(base)
                else:
                    if base in inf:
                        inf.remove(base)
            #Checkbox pour afficher et enlever les données dites civiles complémentaires
            with col2:
                if st.checkbox('Civils comp.', False):
                    inf.extend(details_inf)
                else:
                    if details_inf in inf:
                        inf.remove(details_inf)
            #Checkbox pour afficher et enlever les données dites de revenus complémentaires
            with col3:
                if st.checkbox('Inf. revenus', False):
                    inf.extend(details_perc)
                else:
                    if details_perc in inf:
                        inf.remove(details_perc)
            #Filtrage directement des variables à afficher
            info = st.multiselect("Sélection détaillée des informations à afficher", all, default = inf)
            st.write(data_test[info].loc[data_test['SK_ID_CURR']==ide])

##### Partie Affichage Score et prédiction
        if st.sidebar.checkbox('Score et prédiction', False):
            st.markdown("<h2 style='text-align: center;'>Score et prédiction du client: {}</h2>".format(ide), unsafe_allow_html=True)
      #Client accepté
            if  resultat['prediction'] == 0:
                st.markdown("<h2 style='text-align: center; color: green;'>APPROUVE</h2>", unsafe_allow_html=True)
                if resultat['score'] == 'A':
                    st.image('images/Score-A.png')
                elif resultat['score'] == 'B':
                    st.image('images/Score-B.png')
                else:
                    pass
        #Client refusé
            elif resultat['prediction'] == 1:
                st.markdown("<h2 style='text-align: center; color: red;'>REFUSE</h2>", unsafe_allow_html=True)
                if resultat['score'] == 'C':
                    st.image('images/Score-C.png')
                elif resultat['score'] == 'D':
                    st.image('images/Score-D.png')
                else:
                    pass
            else:
                pass
##### Points fort et points faibles
            with st.container():
                plus, moins, df_local = shap_local(ide)
                col1, col2 = st.columns(2)
        #Points forts clients
                with col1:
                    data_df = pd.DataFrame(
                      {
                        "points forts": plus,
                      }
                    )
                    st.data_editor(
                      data_df,
                      column_config={
                        'points forts': st.column_config.TextColumn(
                          "Points forts",
                          help="Streamlit **widget** commands 🎈",
                          default="st.",
                          max_chars=50,
                          validate="^st\.[a-z_]+$",
                        )
                      },
                      hide_index=True,
                    )
                # Points faibles clients
                with col2:
                    data_df = pd.DataFrame(
                      {
                        "points faibles": moins,
                      }
                    )
                    st.data_editor(
                      data_df,
                      column_config={
                        'points faibles': st.column_config.TextColumn(
                          "Points faibles",
                          help="Streamlit **widget** commands 🎈",
                          default="st.",
                          max_chars=50,
                          validate="^st\.[a-z_]+$",
                        )
                      },
                      hide_index=True,
                    )

#Affichage features importances
            if st.checkbox('Afficher graphe', False):
              df_glob = shap_glob()
              shapey_display(df_glob,df_local)

##### Partie Comparaison
        if st.sidebar.checkbox('Comparaison', False):
            st.markdown("<h2 style='text-align: center;'>Comparaison du client: {}</h2>".format(ide), unsafe_allow_html=True)

 ##### Partie Quantitative 

            if st.sidebar.checkbox('Quantitative', False):
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        disp = st.selectbox('Graphe 1', dict_feat.keys(), index=0, key=1)
                        kde_display(data_test, data_test, dict_feat[disp], ide)

                    with col2:
                        disp_3 = st.selectbox('Graphe 2', dict_feat.keys(), index=3, key=3)
                        kde_display(data_comp, data_test, dict_feat[disp_3], ide)
                if st.checkbox('Afficher plus de graphe:', False, key=123):
                    with st.container():
                        col_1, col_2 = st.columns(2)
                        with col_1:
                            disp_2 = st.selectbox('Graphe 3', dict_feat.keys(), index=2, key=2)
                            kde_display(data_comp, data_test, dict_feat[disp_2], ide)
                        with col_2:
                            disp_4 = st.selectbox('Graphe 4', dict_feat.keys(), index=4, key=4)
                            kde_display(data_comp, data_test, dict_feat[disp_4], ide)
##### Partie Catégorie

            if st.sidebar.checkbox('Catégorie', False):
                with st.container():
                    col1, col2 = st.columns(2)

                    with col1:
                        cat_0 = st.selectbox('Graphe 1', dict_cat.keys(), index=0, key=0)
                        selec = st.radio("Afficher", ['Countplot','Pie chart'], horizontal= True, key=5)
                        pie_bar_display(data_comp, data_test, dict_cat[cat_0], ide, selec)
                    with col2:
                        cat_1 = st.selectbox('Graphe 2', dict_cat.keys(), index=1, key=7)
                        selec_1 = st.radio("Afficher", ['Countplot','Pie chart'], horizontal= True, key=6)
                        pie_bar_display(data_comp, data_test, dict_cat[cat_1], ide, selec_1)
                if st.checkbox('Afficher plus de graphe:', False, key=972):
                    with st.container():
                        col_1, col_2 = st.columns(2)
                        with col_1:
                            cat_2 = st.selectbox('Graphe 3', dict_cat.keys(), index=2, key=8)
                            selec_3 = st.radio("Afficher", ['Countplot','Pie chart'], horizontal= True, key=9)
                            pie_bar_display(data_comp, data_test, dict_cat[cat_2], ide, selec_3)
                        with col_2:
                            cat_3 = st.selectbox('Graphe 4', dict_cat.keys(), index=3, key=10)
                            selec_4 = st.radio("Afficher", ['Countplot','Pie chart'], horizontal= True, key=11)
                            pie_bar_display(data_comp, data_test, dict_cat[cat_3], ide, selec_4)
 ### Partie Bivarié

            if st.sidebar.checkbox('Bivarié', False):
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        bi_x = st.selectbox('Graphe 1, valeur X', dict_feat.keys(), index=2, key=1209)
                        bi_y = st.selectbox('Graphe 1, valeur Y', dict_feat.keys(), index=3, key=190)
                        bi_display(data_test, dict_feat[bi_y2], dict_feat[bi_x2], ide)

                    with col2:
                        bi_x2 = st.selectbox('Graphe 2, valeur X', dict_feat.keys(), index=2, key=3636)
                        bi_y2 = st.selectbox('Graphe 2, valeur Y', dict_feat.keys(), index=5, key=3098)
                        bi_display(data_test, dict_feat[bi_y2], dict_feat[bi_x2], ide)

    st.markdown("Auteur: Stéphane LUBIN")

if __name__ == '__main__':
    main()
