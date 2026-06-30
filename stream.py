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

# Chargement sécurisé du modèle avec joblib
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

    # Création du DataFrame initial
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

    # CORRECTION CRITIQUE : Aligner l'ordre des colonnes sur le modèle
    if hasattr(model, "feature_names_in_"):
        data = data[model.feature_names_in_]

    # Exécution des prédictions
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0]

    st.write("---")
    st.subheader("📈 Résultat de l'analyse")

    if prediction == 1:
        st.error("⚠️ Attaque détectée")
        st.write(f"**Probabilité d'anomalie :** {proba[1]*100:.2f}%")
    else:
        st.success("✅ Pas d'attaque (Trafic Sain)")
        st.write(f"**Probabilité de fiabilité :** {proba[0]*100:.2f}%")

    # -------------------
    # SHAP EXPLANATION (Imbriqué ici pour éviter le NameError)
    # -------------------
    st.write("---")
    st.subheader("🧬 Explication de la décision (SHAP)")

    try:
        # 1. Isoler le modèle et le préprocesseur du Pipeline
        modele_final = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]

        # 2. Transformer les données utilisateur
        transformed_data = preprocessor.transform(data)

        # 3. Récupérer les noms de colonnes post-encodage (ex: OneHotEncoding)
        feature_names = preprocessor.get_feature_names_out()

        # 4. Reconstruire le DataFrame encodé
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

        # 5. Calculer les valeurs SHAP
        explainer = shap.Explainer(modele_final)
        shap_values = explainer(transformed_df)

        # 6. Génération et affichage du graphique en cascade (Waterfall)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)

        # Nettoyage
        plt.clf()
        plt.close(fig)
        
    except Exception as e:
        st.warning("Impossible de charger l'explication SHAP pour le moment.")
        st.info(f"Détail technique : {e}")