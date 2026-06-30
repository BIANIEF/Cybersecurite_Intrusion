import os
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Gestion dynamique du chemin du modèle
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "notebook_des_modeles" / "models" / "random_forest_model.joblib"

# Chargement sécurisé du modèle
model = joblib.load(MODEL_PATH)

st.title("🛡️ Détection d'Intrusion IA")

# -------------------
# INPUT USER
# -------------------
st.subheader("📊 Caractéristiques du trafic réseau")

network_packet_size = st.number_input("Network Packet Size", min_value=0, value=500)
login_attempts = st.number_input("Login Attempts", min_value=0, value=1)
session_duration = st.number_input("Session Duration", min_value=0, value=60)
ip_reputation_score = st.number_input("IP Reputation Score", min_value=0.0, max_value=100.0, value=95.0)
failed_logins = st.number_input("Failed Logins", min_value=0, value=0)

protocol_type = st.selectbox("Protocol Type", ["TCP", "UDP", "ICMP"])
encryption_used = st.selectbox("Encryption Used", ["AES", "DES"])
browser_type = st.selectbox("Browser Type", ["Chrome", "Firefox", "Edge", "Safari"])
unusual_time_access = st.selectbox("Unusual Time Access", ["Yes", "No"])

# -------------------
# PREDICTION & EXPLICATION
# -------------------

if st.button("🔍 Prédire"):

    # 1. Création du DataFrame initial avec les valeurs brutes
    data = pd.DataFrame([{
        "network_packet_size": network_packet_size,
        "protocol_type": protocol_type,
        "login_attempts": login_attempts,
        "session_duration": session_duration,
        "encryption_used": encryption_used,
        "ip_reputation_score": ip_reputation_score,
        "failed_logins": failed_logins,
        "browser_type": browser_type,
        "unusual_time_access": unusual_time_access
    }])

    # Copie pour appliquer les transformations numériques
    data_encoded = data.copy()

    # Si le modèle attend la variable binaire "unusual_time_access" sous forme 0/1
    if hasattr(model, "feature_names_in_") and "unusual_time_access" in model.feature_names_in_:
        data_encoded["unusual_time_access"] = data_encoded["unusual_time_access"].map({"Yes": 1, "No": 0})

    # 2. Application du One-Hot Encoding (génère les colonnes protocol_type_TCP, etc.)
    # dtype=int force l'écriture en 0/1 plutôt qu'en True/False (mieux digéré par Scikit-Learn)
    data_encoded = pd.get_dummies(data_encoded, dtype=int)

    # 3. ALIGNEMENT DYNAMIQUE CRITIQUE
    if hasattr(model, "feature_names_in_"):
        # On ajoute à 0 toutes les colonnes requises par le modèle mais absentes de l'input (ex: browser_type_Unknown)
        for col in model.feature_names_in_:
            if col not in data_encoded.columns:
                data_encoded[col] = 0
        
        # On ordonne et filtre le DataFrame pour correspondre EXACTEMENT aux exigences du modèle
        data_final = data_encoded[model.feature_names_in_]
    else:
        data_final = data_encoded

    # Exécution des prédictions sur les données encodées et alignées
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
    # SHAP EXPLANATION
    # -------------------
    st.write("---")
    st.subheader("🧬 Explication de la décision (SHAP)")

    try:
        # Si le modèle est un Pipeline, on isole le sous-modèle final
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            modele_final = model.named_steps["model"]
        else:
            modele_final = model

        # Calcul des valeurs SHAP directement sur les données finalisées
        explainer = shap.Explainer(modele_final)
        shap_values = explainer(data_final)

        # CORRECTION CRITIQUE : Sélectionner l'échantillon 0 ET la classe prédite
        # Structure de shap_values : [index_echantillon, index_variable, index_classe]
        classe_a_expliquer = int(prediction)  # Prends 0 (Sain) ou 1 (Attaque) dynamiquement
        shap_values_single = shap_values[0, :, classe_a_expliquer]

        # Génération graphique
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values_single, show=False)
        plt.tight_layout()
        st.pyplot(fig)

        # Libération de la mémoire
        plt.clf()
        plt.close(fig)
        
    except Exception as e:
        st.warning("Graphique SHAP indisponible pour cette prédiction.")
        st.info(f"Détail technique SHAP : {e}")