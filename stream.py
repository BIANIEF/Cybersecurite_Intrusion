import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# chargement modèle
with open("modele_intrusion.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🛡️ Détection d'Intrusion IA")

# -------------------
# INPUT USER
# -------------------

network_packet_size = st.number_input("Network Packet Size", min_value=0)
login_attempts = st.number_input("Login Attempts", min_value=0)
session_duration = st.number_input("Session Duration", min_value=0)
ip_reputation_score = st.number_input("IP Reputation Score", min_value=0.0, max_value=100.0)
failed_logins = st.number_input("Failed Logins", min_value=0)

protocol_type = st.selectbox("Protocol Type", ["TCP", "UDP", "ICMP"])
encryption_used = st.selectbox("Encryption Used", ["Yes", "No"])
browser_type = st.selectbox("Browser Type", ["Chrome", "Firefox", "Edge", "Safari"])
unusual_time_access = st.selectbox("Unusual Time Access", ["Yes", "No"])

# -------------------
# PREDICTION
# -------------------

if st.button("🔍 Prédire"):

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

    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0]

    st.subheader("Résultat")

    if prediction == 1:
        st.error("⚠️ Attaque détectée")
        st.write(f"Probabilité : {proba[1]*100:.2f}%")
    else:
        st.success("✅ Pas d'attaque")
        st.write(f"Probabilité : {proba[0]*100:.2f}%")

# -------------------
# SHAP EXPLANATION
# -------------------

st.subheader("Explication (SHAP)")

# 1. Isoler le modèle et le préprocesseur
modele_final = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]

# 2. Transformer les données
transformed_data = preprocessor.transform(data)

# 3. Récupérer les noms de colonnes générés (surtout après un OneHotEncoding)
feature_names = preprocessor.get_feature_names_out()

# 4. Reconstruire un DataFrame avec les bons noms
transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

# 5. Créer l'explicateur et calculer les valeurs SHAP
explainer = shap.Explainer(modele_final)
shap_values = explainer(transformed_df)

# 6. Afficher le graphique proprement
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# Nettoyer la figure de la mémoire
plt.clf()
plt.close(fig)