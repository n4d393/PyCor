import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


##############################################################################
# Fonctions de preprocessing
def organize_dates(data, dates_columns):  
    for col in dates_columns:
        data[col] = pd.to_datetime(data[col])
        data[f'{col}_year']  = data[col].dt.year
        data[f'{col}_month'] = data[col].dt.month
        data[f'{col}_day_name'] = data[col].dt.day_name()
        data[f'{col}_is_weekend'] = np.where(data[f'{col}_day_name'].isin(['Saturday', 'Sunday']), 1, 0)
    return data.drop(dates_columns, axis=1)

def replace_bool(data, boolean_cols): 
    for col in boolean_cols:
        data[col] = np.where(data[col]==True, 1, 0)
    return data

def replace_brand(data, brand_positionning): 
    data['hotel_brand_positionning'] = np.where(data['hotel_brand_positionning'] == 'Unknown Positioning', data['hotel_brand_code'], data['hotel_brand_positionning'])
    data['hotel_brand_positionning'].replace(brand_positionning, inplace=True)
    return data

def preprocessing_data(data, dates_columns, boolean_cols, drop_columns_na, drop_columns):
    print("**** PREPROCESSING *****")
    print("Organize dates")
    data = organize_dates(data, dates_columns)
    print("Replace booleans")
    data = replace_bool(data, boolean_cols)
    print("Add gap stay night")
    data['gap_bkg_stay_nights'] = data['bkg_nbroomnights'] - data['stay_nbroomnights']
    print("Replace brand positionning")
    data = replace_brand(data, brand_positionning)
    print("Drop colums NA") 
    data = data.drop(drop_columns_na, axis=1)
    print("Drop colums") 
    data = data.drop(drop_columns, axis=1)
    print("Dropna")
    data = data.dropna()
    print("Drop None target")
    data = data[data['declared_stay_type']!='None']
    return data
##############################################################################

# Colonnes a supprimer car na
drop_columns_na = [
    'initialmaincro', 'initialsubcro', 'finalmaincro', 'final_subcro', 'bad_tr_ nb_nights', 
    'bad_tr_ turn_over_eur', 'rcu_codemarque', 
    'hotel_zip_code', 'flag_is_ota', 'bad_tr_nbroomnights', 'flag_aberrante_value',
    'bkg_catotal_eur','stay_cor_catotal_eur','bad_tr_ turn_over_eur',
    'flag_aberrante_value','bkg_caroom_eur','stay_caroom_eur','stay_cor_caroom_eur','sb_caroom_eur',
    'nb_child',
    'bkg_nbnights','stay_nbnights',
    'bkg_nbroomnights','stay_nbroomnights'
]

# Colonnes a supprimer car pas de correlation ou doublon
drop_columns = [ # Drop the too specific values not interesting for us
    'hotelcode', 'hotel_name', 'hotel_brand_code', 'hotel_country','hotel_city', 'hotel_country_code',
    'sb_catotal_eur','tr_turn_over_eur','checkout_date_index','sb_nbnights','tr_nbnights','sb_nbroomnights',
    'tr_nbroomnights','calculated_stay_type'
]

# Colonnes de date a formater
dates_columns = ['booking_date', 'checkin_date', 'checkout_date']

# Colonnes Boolean a formater
boolean_columns = ['is_web_direct', 'child_presence', 'eligible_to_earn_points']

# Positionnement des marques
brand_positionning = {  
    'IBS' : 'Economy',
    'BKF' : 'Economy',
    'IBH' : 'Economy',
    'IBB' : 'Economy',
    'GRE' : 'Economy',
    'ADG' : 'Economy',
    'HOF' : 'Economy',
    'NOV' : 'Midscale', 
    'SUI' : 'Midscale',
    'MER' : 'Midscale',
    'SAM' : 'Midscale',
    'AHM' : 'Midscale',
    'TRI' : 'Midscale',
    'MTA' : 'Midscale',
    '21C' : 'Midscale', 
    'MSH' : 'Midscale',
    'BME' : 'Midscale',
    'MOL' : 'Midscale',
    'NOL' : 'Midscale',
    'MGS' : 'Luxury and Upscale',  
    'ADA' : 'Luxury and Upscale',  
    'SOL' : 'Luxury and Upscale', 
    'SEB' : 'Luxury and Upscale', 
    'MGA' : 'Luxury and Upscale', 
    'SOF' : 'Luxury and Upscale',
    'FAI' : 'Luxury and Upscale',
    'SWI' : 'Luxury and Upscale',
    'PUL' : 'Luxury and Upscale',
    'RAF' : 'Luxury and Upscale',
    'ART' : 'Luxury and Upscale',
    'CAS' : 'Luxury and Upscale',
    'MOV' : 'Luxury and Upscale',
    'MEI' : 'Luxury and Upscale',
    'ANG' : 'Luxury and Upscale',
    'PEP' : 'Luxury and Upscale',
    'SO'  : 'Luxury and Upscale',
    'SO/' : 'Luxury and Upscale',
    'SOR' : 'Luxury and Upscale',
    'SWL' : 'Luxury and Upscale',
    'MEL' : 'Luxury and Upscale',
    'TOR' : 'Luxury and Upscale',
    'FAE' : 'Luxury and Upscale'
}

df=pd.read_csv("Fichier_Projet_DS_new_light.csv", sep= ';')
#df=pd.read_csv("Fichier_Projet_DS_new.csv", sep= ';')

st.set_page_config(
    page_title="PyCor",
    page_icon="🧊"
)

st.markdown("<h2 style='text-align: center; color: grey;'>PyCor - Classification des séjours en hôtellerie</h2>", unsafe_allow_html=True)

st.sidebar.title("Sommaire")
pages=["Description du projet", "Le jeu de données" , "Datavisualisation" , "Préparation des données" , "Modélisation" , "Conclusion et perspective"]
page=st.sidebar.radio(" ", pages)
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write("DATASCIENTEST")
st.sidebar.write("Projet DS - promo Sept 2022")
link='[Nadège ACHTERGAELE](https://www.linkedin.com/in/nadege-achtergaele/)'
st.sidebar.markdown(link,unsafe_allow_html=True)
link2='[JB PINEDE](https://www.linkedin.com/in/jean-baptiste-pinede-04a564143/)'
st.sidebar.markdown(link2,unsafe_allow_html=True)

# Description du projet
if page == pages[0] : 
    st.write(' ')
    st.write(' ')
    st.info("### Description du projet")
    st.write(' ')
    st.write(' ')
    st.image("Capture_Reservation_Sejour_sans_adresse_et_tel.JPG")
    st.write(' ')
    st.write(' ')
    st.caption('Les hôtels et le service marketing ont besoin de connaître si les clients réservent dans un hôtel pour un objectif de Leisure ou de Business.')
    st.caption("En effet, l'offre de services ainsi que les campagnes de publicités ne sont absolument pas les mêmes selon la cible visée.")
    st.caption('L’objectif de notre projet est de réussir à **prédire le type de séjour pour chaque réservation faite** quelque soit le canal de distribution utilisé car cette information est facultative.')
    

# Jeu de données
if page == pages[1] : 
    st.write(' ')
    st.write(' ')
    st.info("### Le jeu de données")
    st.write(' ')
    st.caption("Les données ont été prélevées dans le Datalake Accor qui contient l’ensemble des informations liées aux réservations des clients sans aucune mise en forme préalable.")
    st.caption("Le DataSet de départ est donc un fichier .csv de 500 000 lignes sur 65 colonnes anonymisées.")
    st.write(' ')
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.caption("Pour les besoins de la démonstration, nous n'avons pas chargé le fichier complet.")
    st.write(' ')
    st.dataframe(df.describe())
    st.caption("Les données semblent cohérentes. Certains valeurs semblent disproportionnées (par exemple un stay_length de 205 jours ou une réservation pour 16 adultes). Pourtant elles peuvent toutes s'expliquer côté métier donc nous avons décidé de les garder.")
    st.write(' ')
    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

# Datavisualisation
if page == pages[2] :
    st.write(' ')
    st.write(' ')
    st.info("### Datavisualisation")
    st.write(' ')

    st.caption("Comme on peut le voir, nous avons beaucoup de types de séjour non renseignés. On note aussi un déséquilibre entre les données Leisure / Business.") 
    st.write(' ')

    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x = 'declared_stay_type', data = df)        
    plt.title("Distribution des séjours d'après le DataSet")
    st.pyplot(fig)	
	
    st.write(' ')
    st.caption("On peut remarquer qu'en séjour Business, le nombre d'adulte est majoritairement de 1 personne. On peut donc en déduire qu'il s'agit d'une valeur très influente.") 
    st.caption("Le mois du séjour semble aussi intéressant puisque les séjours Leisure se concentrent en période estivale.") 
    st.write(' ')
    col1, col2 = st.columns(2)
	
    with col1:
    	st.image("Nb_Adults.JPG")  	

    with col2:
    	st.image("Stay_type_by_month.JPG")
    
	
    #sns.countplot(x = 'checkin_date_month', hue='declared_stay_type', data = df)
    #st.pyplot(fig)
	
    st.write(' ')
    st.write(' ')

	
    st.caption("### Matrices de corrélation :")    
	
    #fig, ax = plt.subplots()
    #sns.heatmap(df.corr(), ax=ax)
    #st.write(fig)

    # On supprime les lignes ou on a pas de valeur et la colonne calculé
    df_corrnum = df[df['declared_stay_type']!='None']
    df_corrnum = df_corrnum.drop(drop_columns_na, axis=1)
    df_corrnum = df_corrnum.drop(drop_columns, axis=1)

    # On remplace les categories par des variables numeriques
    df_corrnum = df_corrnum.replace(['business'], 1)
    df_corrnum = df_corrnum.replace(['leisure'], 0)

    # Extraction des variables numeriques
    X_num = df_corrnum.select_dtypes(include=['int64','float64','bool','int32'])
    fig, ax = plt.subplots(figsize=(20,10))
    sns.heatmap(X_num.corr(), annot= False, cmap= 'bwr',vmin= -1, vmax= 1, square= True, linewidths= 0.5);
    st.write(fig)

    st.write(' ')
    st.caption("### Interprétabilité") 
    st.caption("On retrouve les variables qui nous semblaient intuitivement incontournables pour la prédiction recherchée.") 
    st.write(' ')
    col1, col2 = st.columns(2)
	
    with col1:
    	st.image("shap_summary_plot.JPG")  	

    with col2:
    	st.image("fet_importance.JPG")

# Préparation des données
if page == pages[3] : 
    st.write(' ')
    st.write(' ')

    # Preprocessing
    st.info("### Préparation des données")
    st.write(' ')
    st.write(' ')
    st.image("SchemaPreproc.png")
    st.write(' ')
    st.caption("### 1) Séparation des features et de la target")
    st.caption("### 2) Dichotomisation des features et numérisation de la target : dummies")
    st.caption("### 3) Séparation de notre jeu en 2 parties : train_test_split")
    st.caption("### 4) Normalisation des features : StandardScaler")
    st.caption("### 5) Sélection des meilleures features : SelectKBest")

# Modélisation
if page == pages[4]:
    # Modelisation
    st.info("### Modélisation")
    st.caption("### Objectifs de résultat de notre modèle : Accuracy > 0.8")
    st.caption("Le modèle devra obtenir un taux de précision minimale de 0.8 globalement, et ensuite si possible de privilégier la classe « Business »")
    st.caption("### Test de 3 modèles de classification :")
    st.caption("- Logistic regression : Modèle simple")
    st.caption("- Random Forest : Bagging sur les arbres de décision")
    st.caption("- XGBoost : Gradient Boosting")

    st.caption("### Optimisation par GridSearchCV")
    
    # Preproc
    #df_preproc = preprocessing_data(df, dates_columns, boolean_columns, drop_columns_na, drop_columns)
    #df_preproc.to_csv("Fichier_Projet_DS_new_light_preproc.csv", sep= ';') 
    df_preproc = pd.read_csv("Fichier_Projet_DS_new_light_preproc.csv", sep= ';')

    # Creation des features et de la target
    features = df_preproc.drop('declared_stay_type', axis = 1)
    target = df_preproc['declared_stay_type']

    # Remplacement dans Target par des valeurs numeriques
    target = target.replace(['business'], 1)
    target = target.replace(['leisure'], 0)

    # Dummification
    features = pd.get_dummies(features)

    # Creation des jeux d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state=321)

    # Standardisation des valeurs
    scaler= StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)

    def prediction(classifier):
        if classifier == 'Logistic Regression':
            #clf = LogisticRegression()
            #clf.fit(X_train, y_train)
            #joblib.dump(clf, "LogiscticRegression") 
            clf = joblib.load("LogiscticRegression") 
        elif classifier == 'Random Forest':
            #clf = RandomForestClassifier()
            #clf.fit(X_train, y_train)
            #joblib.dump(clf, "RandomForest")
            clf = joblib.load("RandomForest")
        elif classifier == 'XGBoost':
            #eval_set = [(X_train, y_train),(X_test,y_test)]
            #clf = xgb.XGBClassifier(
            #booster="gbtree",
            #subsample=1,
            #colsample_bytree=1,
            #min_child_weight=1,
            #max_depth=6,
            #learning_rate=0.3,
            #n_estimators=100,
            #eval_metric="error",
            #early_stopping_rounds=10)
            #clf.fit(X_train, y_train, eval_set=eval_set, verbose=0)
            #joblib.dump(clf, "XGBoost")
            clf = joblib.load("XGBoost")
        elif classifier == 'XGBoost GridSearchCV':
            #eval_set = [(X_train, y_train),(X_test,y_test)]
            #clf = xgb_classifier = xgb.XGBClassifier(
            #booster="gbtree",
            #subsample=1,
            #colsample_bytree=1,
            #min_child_weight=1,
            #max_depth=12,
            #learning_rate=0.1,
            #n_estimators=200,
            #eval_metric="error",
            #early_stopping_rounds=10)
            #xgb_classifier.fit(X_train, y_train, eval_set=eval_set, verbose=0)
            #joblib.dump(clf, "XGBoostGridSearchCV")
            clf = joblib.load("XGBoostGridSearchCV")
        return clf
	
    def scores(clf, y_pred, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
        
    choix = ['Logistic Regression','Random Forest', 'XGBoost', "XGBoost GridSearchCV"]
    option = st.selectbox('Choix du modèle', choix)
    st.write('Résultats du modèle ', option)

    clf = prediction(option)

    #display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    #if display == 'Accuracy':
    #    st.write(scores(clf, display))
    #elif display == 'Confusion matrix':
    #    st.dataframe(scores(clf, display))


    y_pred = clf.predict(X_test)
    st.write("Accuracy : ", scores(clf, y_pred, 'Accuracy'))
    st.text("Classification report :\n" + classification_report(y_true= y_test, y_pred=y_pred))
    st.write("Matrice de confusion (0) Leisure / (1) Business : ", scores(clf, y_pred, 'Confusion matrix'))
	
# Pour aller plus loin
if page == pages[5] : 
    st.write(' ')
    st.write(' ')
    # Pour aller plus loin
    st.info("### Conclusion")
    st.caption("Le modèle XGBoost est le plus performant et répond au besoin (Accuracy de 0.83), cependant la classe Business est la moins bien prédite.")

    st.info("### Pour aller plus loin : Optimisation de la classe Business")
    st.caption("Nous avons essayé d'utiliser une RFE (Recursive Feature Elimination) avec un scorer personnalisé sur la classe Business pour augmenter son score. Mais cela n'a pas fonctionné et a juste diminué le score de la classe Leisure.")
    st.caption("Piste à explorer pour améliorer la performance :")
    st.caption("- Rééquilibrage du jeu de données entre les deux classes") 
    st.caption("- Utiliser un modèle semi supervisé") 
