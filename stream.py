import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Chargement
model = joblib.load("notebook_des_modeles/models/random_forest_model.joblib")
scaler = joblib.load("notebook_des_modeles/models/scaler.joblib")

st.title("🛡️ Détection d'Intrusion IA")

# ... (tes inputs restent identiques) ...

if st.button("🔍 Prédire"):
    # ... (ton code de preprocessing reste identique) ...

    # Calcul
    prediction = model.predict(data_final)[0]
    proba = model.predict_proba(data_final)[0]

    st.subheader("📈 Résultat")
    if prediction == 1:
        st.error(f"⚠️ Attaque détectée ({proba[1]*100:.1f}%)")
    else:
        st.success(f"✅ Trafic Sain ({proba[0]*100:.1f}%)")

    # Visualisation des variables influentes (sans texte complexe)
    st.write("---")
    st.subheader("📊 Facteurs d'influence")
    
    # Calcul simplifié de l'importance
    importances = pd.Series(model.feature_importances_, index=data_final.columns)
    top_features = importances.sort_values(ascending=False).head(5)
    
    fig, ax = plt.subplots()
    top_features.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title("Top 5 des variables influentes pour ce modèle")
    st.pyplot(fig)