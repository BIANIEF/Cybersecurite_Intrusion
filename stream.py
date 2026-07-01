import os
import numpy as np
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# BASE_DIR est déjà le dossier racine (CYBERSECURITE_INTRUSION)
BASE_DIR = Path(__file__).resolve().parent

# Chemin explicite vers le sous-dossier
MODEL_PATH = BASE_DIR / "notebook_des_modeles" / "models" / "random_forest_model.joblib"
SCALER_PATH = BASE_DIR / "notebook_des_modeles" / "models" / "scaler.joblib"

# Chargement
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Chargement sécurisé
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) # <-- Le scaler utilisé pendant l'entraînement

st.title("🛡️ Détection d'Intrusion IA")

# -------------------
# INPUT USER
# -------------------
st.subheader("📊 Caractéristiques du trafic réseau")

network_packet_size = st.number_input("Network Packet Size", min_value=0, value=500)
login_attempts = st.number_input("Login Attempts", min_value=0, value=1)
session_duration = st.number_input("Session Duration (secondes)", min_value=0, value=60)
ip_reputation_score = st.number_input("IP Reputation Score", min_value=0.0, max_value=100.0, value=95.0)
failed_logins = st.number_input("Failed Logins", min_value=0, value=0)

protocol_type = st.selectbox("Protocol Type", ["TCP", "UDP", "ICMP"])
encryption_used = st.selectbox("Encryption Used", ["AES", "DES", "None"])
browser_type = st.selectbox("Browser Type", ["Chrome", "Firefox", "Edge", "Safari", "Unknown"])
unusual_time_access = st.selectbox("Unusual Time Access", ["Yes", "No"])

# -------------------
# PREDICTION & EXPLICATION
# -------------------

if st.button("🔍 Prédire"):

    # 1. Création du DataFrame initial avec les valeurs brutes
    data = pd.DataFrame([{
        "network_packet_size": network_packet_size,
        "login_attempts": login_attempts,
        "session_duration": session_duration,
        "ip_reputation_score": ip_reputation_score,
        "failed_logins": failed_logins,
        "protocol_type": protocol_type,
        "encryption_used": encryption_used,
        "browser_type": browser_type,
        "unusual_time_access": unusual_time_access
    }])

    data_encoded = data.copy()

    # --- DEBUT DU PREPROCESSING EXACT ---
    
    # A. Transformation Logarithmique (comme dans le train)
    data_encoded['session_duration'] = np.log1p(data_encoded['session_duration'])

    # B. Conversion de la variable booléenne
    data_encoded["unusual_time_access"] = data_encoded["unusual_time_access"].map({"Yes": 1, "No": 0})

    # C. Encodage One-Hot des variables catégorielles
    data_encoded = pd.get_dummies(data_encoded, columns=['protocol_type', 'encryption_used', 'browser_type'], dtype=int)

    # D. Alignement dynamique des colonnes avec celles du modèle
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in data_encoded.columns:
                data_encoded[col] = 0
        
        # On s'assure d'avoir l'ordre exact attendu par le modèle
        data_final = data_encoded[model.feature_names_in_].copy()
    else:
        data_final = data_encoded.copy()

    # E. Normalisation MinMax sur les colonnes ciblées (avec le scaler sauvegardé)
    cols_to_scale = ['network_packet_size', 'session_duration', 'login_attempts']
    # On vérifie que les colonnes sont bien présentes avant de scale
    if all(col in data_final.columns for col in cols_to_scale):
        data_final[cols_to_scale] = scaler.transform(data_final[cols_to_scale])

    # --- FIN DU PREPROCESSING ---

    # Exécution des prédictions
    prediction = model.predict(data_final)[0]
    proba = model.predict_proba(data_final)[0]

    st.write("---")
    st.subheader("📈 Résultat de l'analyse")

    if prediction == 1:
        st.error("⚠️ Attaque détectée")
        st.write(f"**Probabilité d'anomalie :** {proba[1]*100:.2f}%")
    else:
        st.success("✅ Pas d'attaque (Trafic Sain)")
        st.write(f"**Probabilité de fiabilité :** {proba[0]*100:.2f}%")

    # -------------------
    # EXPLICATION TEXTUELLE (SHAP)
    # -------------------
    st.write("---")
    st.subheader("📝 Rapport d'analyse de l'IA")

    try:
        # Optimisation : Utilisation de TreeExplainer plus performant pour Random Forest
        explainer = shap.TreeExplainer(model)
        
        # SHAP renvoie parfois des listes selon la version, on gère les dimensions
        shap_values = explainer.shap_values(data_final)
        
        # Sur un Random Forest binaire, shap_values est souvent une liste [sain, attaque]
        if isinstance(shap_values, list):
            shap_values_single = shap_values[int(prediction)][0]
        else:
            shap_values_single = shap_values[0]

        importance_df = pd.DataFrame({
            'Variable': data_final.columns,
            'Impact': shap_values_single,
            'Valeur_Saisie': data_final.iloc[0].values
        })

        importance_df['Impact_Absolu'] = importance_df['Impact'].abs()
        importance_df = importance_df.sort_values(by='Impact_Absolu', ascending=False)
        top_3 = importance_df.head(3)

        statut_prediction = "DÉTECTION D'UNE INTRUSION" if prediction == 1 else "TRAFIC LÉGITIME"
        classe_a_expliquer = int(prediction)
        
        paragraphe = f"""
        L'algorithme a classé cet événement comme un **{statut_prediction}** avec un niveau de confiance de **{proba[classe_a_expliquer]*100:.1f}%**. 
        Après analyse approfondie des caractéristiques du signal, cette décision s'explique principalement par la combinaison de trois facteurs majeurs :
        """
        st.write(paragraphe)

        for _, row in top_3.iterrows():
            nom_var = row['Variable'].replace('_', ' ').title()
            valeur = round(row['Valeur_Saisie'], 4) if isinstance(row['Valeur_Saisie'], float) else row['Valeur_Saisie']
            
            if row['Impact'] > 0:
                action = "a fortement **conforté** l'évaluation du modèle dans son choix actuel"
            else:
                action = "a agi comme un contre-poids, mais n'a pas suffi à inverser la tendance"

            st.write(f"* **{nom_var}** (Valeur traitée : `{valeur}`) : Ce paramètre {action} (poids SHAP : `{row['Impact']:.3f}`).")

    except Exception as e:
        st.warning("Génération du rapport textuel indisponible pour cette prédiction.")
        st.info(f"Détail technique : {e}")