import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import logging
from datetime import datetime
warnings.filterwarnings("ignore")

logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Authentification

if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
 
if not st.session_state.authentifie:
    st.title("Health-InsurTech - Connexion")
    st.markdown("---")
    identifiant  = st.text_input("Identifiant")
    mot_de_passe = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if identifiant == "admin" and mot_de_passe == "admin":
            st.session_state.authentifie = True
            logging.info(f"Connexion reussie - utilisateur : {identifiant}")
            st.rerun()
        else:
            logging.warning(f"Tentative de connexion echouee - identifiant : {identifiant}")
            st.error("Identifiant ou mot de passe incorrect")
    st.stop()
 
st.set_page_config(page_title="Health-InsurTech", layout="wide")
 
 
# Consentement RGPD
if "consentement" not in st.session_state:
    st.session_state.consentement = False
 
if not st.session_state.consentement:
    st.title("Health-InsurTech")
    st.warning("### Avant de continuer")
    st.write("""
    Cette application utilise vos données de santé (âge, IMC, statut fumeur)
    pour estimer vos frais médicaux annuels.
 
    Conformément au RGPD :
    - Vos données ne sont pas stockées
    - Elles sont traitées uniquement pendant la simulation
    - Vous pouvez fermer l'application à tout moment
    """)
    if st.button("J'accepte et je continue"):
        st.session_state.consentement = True
        st.rerun()
    st.stop()
 
 
# Chargement & entraînement
@st.cache_data
def charger_et_entrainer():
    insurance = pd.read_csv("insurance_data.csv")
 
    colonnes_a_supprimer = [
        "nom", "prenom", "date_naissance", "sexe",
        "email", "telephone", "numero_secu_sociale",
        "ville", "code_postal", "region_fr",
        "adresse_ip", "mutuelle_complementaire",
        "date_inscription", "consentement_rgpd", "sex"
    ]
    colonnes_a_supprimer = [c for c in colonnes_a_supprimer if c in insurance.columns]
 
    insurance_clean = (
        insurance[insurance["consentement_rgpd"] == "Oui"]
        .drop(columns=colonnes_a_supprimer)
        .copy()
    )
 
    insurance_viz = insurance_clean.copy()
 
    insurance_clean = pd.get_dummies(insurance_clean, columns=["smoker", "region"], drop_first=True)
    insurance_clean = insurance_clean.rename(columns={"smoker_yes": "est_fumeur"})
 
    X = insurance_clean.drop(columns=["id_client", "charges"])
    y = insurance_clean["charges"]
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
 
    r2 = r2_score(y_test, rf.predict(X_test))
 
    return rf, X.columns.tolist(), r2, insurance_viz
 
rf, feature_names, r2, insurance_viz = charger_et_entrainer()

# Sidebar
st.sidebar.title("Appli Health-Insur Tech")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Simulation tarif"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Modele : Random Forest | R2 = {r2:.4f}")

# Page Dashboard

if page == "Dashboard":
    st.title("Analyse des frais medicaux interactif")
    st.markdown(f"**{len(insurance_viz)} clients** | Donnees anonymisees | RGPD conforme")
    st.markdown("---")

    col1, col2, col3 =st.columns(3)
    col1.metric("Frais moyens",  f"{insurance_viz['charges'].mean():,.0f} EUR")
    col2.metric("Frais medians", f"{insurance_viz['charges'].median():,.0f} EUR")
    col3.metric("Fumeurs",       f"{(insurance_viz['smoker'] == 'yes').sum()} clients")
 
    st.markdown("---")

    st.subheader("Répartition des frais médicaux")
    fig_dist = px.histogram(
        insurance_viz,
        x="charges",
        nbins=50,
        color="smoker",
        color_discrete_map={"yes": "#EF553B", "no": "#636EFA"},
        labels={"charges": "Frais medicaux (EUR)", "smoker": "Fumeur"},
        barmode="overlay",
        opacity=0.75,
    )
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    st.subheader("Correlation de l'IMC, Age et Frais médicaux")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_bmi = px.scatter(
            insurance_viz,
            x="bmi", y="charges", color="smoker",
            color_discrete_map={"yes": "#EF553B", "no": "#636EFA"},
            labels={"bmi": "IMC", "charges": "Frais (EUR)", "smoker": "Fumeur"},
            title="IMC vs Frais medicaux",
            trendline="ols",
            opacity=0.6,
        )
        fig_bmi.update_layout(height=400)
        st.plotly_chart(fig_bmi, use_container_width=True)
 
    with col_g2:
        fig_age = px.scatter(
            insurance_viz,
            x="age", y="charges", color="smoker",
            color_discrete_map={"yes": "#EF553B", "no": "#636EFA"},
            labels={"age": "Age", "charges": "Frais (EUR)", "smoker": "Fumeur"},
            title="Age vs Frais medicaux",
            trendline="ols",
            opacity=0.6,
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)

# Page Simulation
elif page == "Simulation tarif":
    st.title("Simulation des frais médicaux")
    st.info("Les données ne sont pas conservées. Seule la simulation les utilise")
    
    st.markdown("---")

    col_form, col_result = st.columns(2)

    with col_form:
        st.subheader("Vos infos")
        age      = st.slider("Age", min_value=18, max_value=64, value=35)
        bmi      = st.slider("IMC", min_value=15.0, max_value=55.0, value=28.0, step=0.1)
        children = st.number_input("Nombre d'enfants a charge", min_value=0, max_value=10, value=0)
        smoker   = st.selectbox("Statut fumeur", options=["Non-fumeur", "Fumeur"])
        region   = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])
        simuler  = st.button("Estimer mes frais", use_container_width=True)


    with col_result:
        st.subheader("Resultat")
        if simuler:
            donnees_client = pd.DataFrame([{
                "age"              : age,
                "bmi"              : bmi,
                "children"         : children,
                "est_fumeur"       : 1 if smoker == "Fumeur" else 0,
                "region_northwest" : 1 if region == "northwest" else 0,
                "region_southeast" : 1 if region == "southeast" else 0,
                "region_southwest" : 1 if region == "southwest" else 0,
            }])[feature_names]

            estimation = rf.predict(donnees_client)[0]

            logging.info(f"Simulation - age={age} bmi={bmi} enfants={children} fumeur={smoker} region={region} estimation={estimation:,.0f}")
 
            st.success(f"Estimation : {estimation:,.0f} EUR/an")
            st.caption(f"Soit environ {estimation/12:,.0f} EUR/mois")
            st.markdown("---")
 
            facteurs = pd.DataFrame({
                "Facteur": ["Age", "IMC", "Enfants", "Statut fumeur", "Region"],
                "Valeur" : [age, bmi, children, smoker, region]
            })
            st.dataframe(facteurs, hide_index=True, use_container_width=True)
            st.caption("Cette estimation reste une simulation. Elle ne constitue pas un quelconque contrat d'assurance.")
 
        else:
            st.write("Renseignez vos informations et cliquez sur Estimer mes frais.")
