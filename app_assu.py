import pandas as pd
import streamlit as st
from joblib import load


def layout():
    st.sidebar.title("Information Essentielle: ")
    st.sidebar.title("PRÉDICTION DES COÛTS D'ASSURANCE MÉDICALE ")
    st.sidebar.write("*Il s'agit d'un projet d'apprentissage automatique de prédiction des coûts d'assurance médicale.*")
    st.sidebar.write(" Réalisée par: Ano N'gozan Louis")
    st.sidebar.title(" @2023")

    

@st.cache_data()
def my_cached_function():
    path_data = 'Xtest_ytest.csv'
    df = pd.read_csv(path_data)
    return df



def parametres():
    age = st.slider('Quel âge as-tu?', 0, 130, 25)
    st.write("J'ai", age, 'ans')

    bmi = st.text_input('(bim) rapport entre votre poids et votre taille', value='25.5')
    st.write("Valeur de bim sélectionnée :", bmi)

    smoker = st.selectbox('smoker: 0 -> Non, 1 -> Oui', ['Non', 'Oui'])
    st.write('Vous avez sélectionné:', smoker)


    return age, bmi, smoker

def load_model():
    mod = load(filename='Assurance_final.joblib')
    return mod

def prediction(mod, data):
    data['smoker'] = data['smoker'].map({'Non': 0, 'Oui': 1})  # Convertir 'Non' en 0 et 'Oui' en 1
    resultat = mod.predict(data)
    return resultat




########################################### MAIN ###########################################################

if __name__ == "__main__":
    st.set_page_config(
        page_title="App Assurance",
        layout="centered"
    )
    st.title("Application Web de prédiction d'assurance médicale")
    
    layout()

    # Utilisation de la nouvelle commande de mise en cache
    cached_data = my_cached_function()



     # Chargement des données brutes
    raw_data = my_cached_function()

     # Bouton à cocher pour afficher/masquer les données brutes
    show_raw_data = st.checkbox("Afficher les données brutes")

    if show_raw_data:
        st.caption('Données brutes du dataset')
        st.write(raw_data)
    
    ############ PARAMETRES ##########################
    age, bmi, smoker = parametres()
    
    st.caption('Ci-dessous - DataFrame d\'informations sur l\'Assurance.')
    
    ############ Données Formuliaire ##########################
    data = {'age': [age], 'bmi': [bmi], 'smoker': [smoker]}
    df_formulaire = pd.DataFrame(data)
    st.write(df_formulaire)
    
    

    # Bouton pour appeler le Modèle
    if st.sidebar.button('Valider la Prédiction'):
        mod = load_model()
        resultat = prediction(mod, df_formulaire)
        st.info(f'Prédiction de Coût d\'Assurance: {resultat}')