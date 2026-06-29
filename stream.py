"""
Application Streamlit - Détection d'attaques réseau
Modèle Random Forest + Explications SHAP (globales et locales)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Détection d'Intrusion Réseau",
    page_icon="🛡️",
    layout="wide"
)

# Colonnes du dataset
NUMERIC_COLS_TO_SCALE = ["network_packet_size", "session_duration", "login_attempts"]
CATEGORICAL_COLS = ["protocol_type", "encryption_used", "browser_type"]
LOG_COL = "session_duration"
TARGET_COL = "attack_detected"
ID_COL = "session_id"

# Valeurs possibles connues pour les catégories (au cas où le formulaire manuel
# doit proposer des choix sans dépendre uniquement du CSV)
DEFAULT_PROTOCOLS = ["TCP", "UDP", "ICMP"]
DEFAULT_ENCRYPTIONS = ["AES", "DES"]
DEFAULT_BROWSERS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]


# ============================================================
# FONCTIONS DE PRÉTRAITEMENT
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes avec NaN dans encryption_used (et autres colonnes critiques)."""
    df = df.copy()
    df = df.dropna(subset=["encryption_used"])
    return df


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Applique log1p sur session_duration."""
    df = df.copy()
    df[LOG_COL] = np.log1p(df[LOG_COL])
    return df


def fit_scaler_on_train(train_df: pd.DataFrame) -> StandardScaler:
    """
    Recalcule un StandardScaler à partir du CSV d'entraînement original,
    en appliquant le même pipeline (clean -> log -> scale) que pendant
    l'entraînement du modèle.
    """
    df = clean_data(train_df)
    df = apply_log_transform(df)
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_COLS_TO_SCALE])
    return scaler


def build_feature_columns(train_df: pd.DataFrame) -> list:
    """
    Détermine la liste finale des colonnes de features (après one-hot encoding)
    en se basant sur le CSV d'entraînement, pour garantir l'alignement des
    colonnes avec ce qu'attend le modèle.
    """
    df = clean_data(train_df)
    df = apply_log_transform(df)
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    feature_cols = [c for c in df_encoded.columns if c not in [TARGET_COL, ID_COL]]
    return feature_cols


def preprocess(df: pd.DataFrame, scaler: StandardScaler, feature_cols: list,
                drop_na: bool = True) -> pd.DataFrame:
    """
    Pipeline complet de prétraitement appliqué à de nouvelles données
    (upload ou saisie manuelle), pour les rendre compatibles avec le modèle :
      1. Suppression des NaN dans encryption_used (si drop_na=True)
      2. Log-transformation de session_duration
      3. Normalisation (StandardScaler déjà entraîné) des colonnes numériques
      4. One-hot encoding des colonnes catégorielles
      5. Alignement des colonnes sur celles utilisées à l'entraînement
    """
    df = df.copy()

    if drop_na:
        df = clean_data(df)

    df = apply_log_transform(df)

    # Normalisation avec le scaler déjà entraîné
    df[NUMERIC_COLS_TO_SCALE] = scaler.transform(df[NUMERIC_COLS_TO_SCALE])

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    # Garder une copie de session_id si présent, avant de le retirer des features
    session_ids = df_encoded[ID_COL] if ID_COL in df_encoded.columns else None

    # Aligner les colonnes sur celles de l'entraînement (ajoute les colonnes
    # manquantes avec des 0, retire les colonnes en trop, garde le bon ordre)
    X = df_encoded.reindex(columns=feature_cols, fill_value=0)

    return X, session_ids


# ============================================================
# CHARGEMENT DU MODÈLE (mis en cache)
# ============================================================

@st.cache_resource
def load_model(uploaded_model_file):
    """Charge le modèle Random Forest depuis un fichier .pkl uploadé."""
    try:
        model = pickle.load(uploaded_model_file)
    except Exception:
        uploaded_model_file.seek(0)
        model = joblib.load(uploaded_model_file)
    return model


@st.cache_data
def load_train_data(uploaded_train_file):
    """Charge le CSV d'entraînement original."""
    return pd.read_csv(uploaded_train_file)


@st.cache_resource
def get_scaler_and_features(train_df):
    """Calcule le scaler et la liste des colonnes de features."""
    scaler = fit_scaler_on_train(train_df)
    feature_cols = build_feature_columns(train_df)
    return scaler, feature_cols


@st.cache_resource
def get_shap_explainer(_model):
    """Crée l'explainer SHAP (TreeExplainer adapté à Random Forest)."""
    return shap.TreeExplainer(_model)


def extract_shap_for_positive_class(shap_values, expected_value):
    """
    Normalise la sortie de explainer.shap_values() / explainer.expected_value
    pour ne garder que la classe positive ("attaque détectée"), quelle que
    soit la version de la librairie shap installée :
      - liste [array_classe_0, array_classe_1]                 (anciennes versions)
      - array 3D (n_samples, n_features, n_classes)             (versions récentes)
      - array 2D (n_samples, n_features)                        (sortie déjà binaire)
    Retourne (shap_array_2D, base_value_scalaire).
    """
    if isinstance(shap_values, list):
        sv = shap_values[1]
        base = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
        base = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
    else:
        sv = shap_values
        base = expected_value[1] if hasattr(expected_value, "__len__") and len(expected_value) > 1 else expected_value

    base = float(np.asarray(base).reshape(-1)[0]) if hasattr(base, "__len__") else float(base)
    return sv, base


# ============================================================
# SIDEBAR - CHARGEMENT DES FICHIERS
# ============================================================

st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("1. Modèle entraîné")
model_file = st.sidebar.file_uploader(
    "Charger le modèle (.pkl)", type=["pkl", "joblib"], key="model_uploader"
)

st.sidebar.subheader("2. Données d'entraînement")
train_file = st.sidebar.file_uploader(
    "Charger train_data.csv (pour recalculer le scaler et les explications globales)",
    type=["csv"], key="train_uploader"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Ces deux fichiers sont nécessaires au fonctionnement de l'application : "
    "le modèle pour prédire, le CSV d'entraînement pour reproduire le même "
    "prétraitement (normalisation, encodage) qu'au moment de l'entraînement."
)

# ============================================================
# TITRE PRINCIPAL
# ============================================================

st.title("🛡️ Détection d'Attaques Réseau avec Explicabilité SHAP")
st.markdown(
    "Cette application utilise un modèle **Random Forest** pour détecter des "
    "sessions réseau suspectes, et **SHAP** pour expliquer les prédictions."
)

if model_file is None or train_file is None:
    st.warning("⬅️ Merci de charger le modèle (.pkl) et le CSV d'entraînement dans la barre latérale pour commencer.")
    st.stop()

# Chargement des ressources
with st.spinner("Chargement du modèle et préparation du prétraitement..."):
    model = load_model(model_file)
    train_df_raw = load_train_data(train_file)
    scaler, feature_cols = get_scaler_and_features(train_df_raw)
    explainer = get_shap_explainer(model)

st.success(f"✅ Modèle chargé avec succès. {len(feature_cols)} features attendues par le modèle.")

# ============================================================
# ONGLETS PRINCIPAUX
# ============================================================

tab_global, tab_upload, tab_manual = st.tabs([
    "🌍 Explication globale (SHAP)",
    "📂 Prédiction par upload CSV",
    "✍️ Prédiction par saisie manuelle"
])

# ------------------------------------------------------------
# ONGLET 1 : EXPLICATION GLOBALE
# ------------------------------------------------------------
with tab_global:
    st.header("Importance globale des variables")
    st.markdown(
        "Cette vue montre quelles variables influencent le plus les prédictions "
        "du modèle, sur l'ensemble des données d'entraînement."
    )

    sample_size = st.slider(
        "Taille de l'échantillon utilisé pour le calcul SHAP (plus petit = plus rapide)",
        min_value=100, max_value=min(2000, len(train_df_raw)), value=min(500, len(train_df_raw)), step=100
    )

    if st.button("🔍 Calculer les explications globales", type="primary"):
        with st.spinner("Calcul des valeurs SHAP en cours..."):
            sample_df = train_df_raw.sample(n=sample_size, random_state=42)
            X_global, _ = preprocess(sample_df, scaler, feature_cols, drop_na=True)

            shap_values_raw = explainer.shap_values(X_global)
            shap_values_plot, _ = extract_shap_for_positive_class(shap_values_raw, explainer.expected_value)

        st.subheader("📊 Importance des variables (summary plot)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_global, show=False)
        st.pyplot(fig1, bbox_inches="tight")
        plt.close(fig1)

        st.subheader("📊 Importance moyenne (bar plot)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_global, plot_type="bar", show=False)
        st.pyplot(fig2, bbox_inches="tight")
        plt.close(fig2)

        st.info(
            "**Lecture du graphique** : chaque point représente une session. "
            "Les variables sont classées par importance décroissante. "
            "Une valeur SHAP positive pousse la prédiction vers 'attaque détectée', "
            "une valeur négative la pousse vers 'pas d'attaque'."
        )

# ------------------------------------------------------------
# ONGLET 2 : PRÉDICTION PAR UPLOAD CSV
# ------------------------------------------------------------
with tab_upload:
    st.header("Prédiction sur un fichier CSV")
    st.markdown(
        "Le fichier doit contenir les mêmes colonnes que le dataset original "
        "(hors colonne `attack_detected`)."
    )

    pred_file = st.file_uploader("Charger un fichier CSV à analyser", type=["csv"], key="pred_uploader")

    if pred_file is not None:
        new_df = pd.read_csv(pred_file)
        st.write("Aperçu des données chargées :")
        st.dataframe(new_df.head())

        n_before = len(new_df)
        new_df_clean = clean_data(new_df)
        n_after = len(new_df_clean)
        if n_after < n_before:
            st.warning(f"⚠️ {n_before - n_after} ligne(s) supprimée(s) à cause de valeurs manquantes dans `encryption_used`.")

        if st.button("🚀 Lancer la prédiction", type="primary"):
            with st.spinner("Prétraitement et prédiction en cours..."):
                X_new, session_ids = preprocess(new_df, scaler, feature_cols, drop_na=True)
                predictions = model.predict(X_new)
                probabilities = model.predict_proba(X_new)[:, 1]

            results_df = new_df_clean.reset_index(drop=True).copy()
            results_df["prediction"] = predictions
            results_df["probabilite_attaque"] = probabilities.round(4)
            results_df["label"] = results_df["prediction"].map({0: "✅ Normal", 1: "🚨 Attaque détectée"})

            st.subheader("Résultats des prédictions")
            st.dataframe(results_df[[ID_COL, "label", "probabilite_attaque"] if ID_COL in results_df.columns
                                     else ["label", "probabilite_attaque"]])

            n_attacks = int(predictions.sum())
            col1, col2 = st.columns(2)
            col1.metric("Sessions analysées", len(predictions))
            col2.metric("Attaques détectées", n_attacks)

            # Stocker en session_state pour l'explication locale
            st.session_state["X_new_upload"] = X_new
            st.session_state["results_df_upload"] = results_df

            csv_export = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger les résultats (CSV)", csv_export,
                                "resultats_predictions.csv", "text/csv")

        # Explication locale pour une ligne du CSV uploadé
        if "X_new_upload" in st.session_state:
            st.markdown("---")
            st.subheader("🔬 Explication locale d'une session")
            X_new = st.session_state["X_new_upload"]
            results_df = st.session_state["results_df_upload"]

            row_idx = st.number_input(
                "Index de la ligne à expliquer (dans le fichier nettoyé)",
                min_value=0, max_value=len(X_new) - 1, value=0, step=1
            )

            if st.button("💡 Expliquer cette session", key="explain_upload_row"):
                with st.spinner("Calcul de l'explication SHAP..."):
                    shap_values_row_raw = explainer.shap_values(X_new.iloc[[row_idx]])
                    sv_2d, base_value = extract_shap_for_positive_class(shap_values_row_raw, explainer.expected_value)
                    sv = sv_2d[0]

                pred_label = results_df.iloc[row_idx]["label"]
                proba = results_df.iloc[row_idx]["probabilite_attaque"]
                st.markdown(f"**Prédiction** : {pred_label} (probabilité d'attaque : {proba:.2%})")

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots._waterfall.waterfall_legacy(
                    base_value, sv, feature_names=X_new.columns.tolist(), show=False
                )
                st.pyplot(fig, bbox_inches="tight")
                plt.close(fig)

                st.info(
                    "**Lecture du graphique** : les barres rouges poussent la prédiction "
                    "vers 'attaque détectée', les barres bleues vers 'normal'. "
                    "Plus la barre est longue, plus l'impact de la variable est important."
                )

# ------------------------------------------------------------
# ONGLET 3 : PRÉDICTION PAR SAISIE MANUELLE
# ------------------------------------------------------------
with tab_manual:
    st.header("Prédiction pour une session unique")
    st.markdown("Renseignez les caractéristiques de la session réseau à analyser.")

    # Récupération des valeurs catégorielles présentes dans le CSV d'entraînement
    protocols = sorted(train_df_raw["protocol_type"].dropna().unique().tolist()) or DEFAULT_PROTOCOLS
    encryptions = sorted(train_df_raw["encryption_used"].dropna().unique().tolist()) or DEFAULT_ENCRYPTIONS
    browsers = sorted(train_df_raw["browser_type"].dropna().unique().tolist()) or DEFAULT_BROWSERS

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            network_packet_size = st.number_input("Taille des paquets réseau (network_packet_size)",
                                                     min_value=0, value=500)
            login_attempts = st.number_input("Nombre de tentatives de connexion (login_attempts)",
                                               min_value=0, value=1)
            failed_logins = st.number_input("Connexions échouées (failed_logins)",
                                              min_value=0, value=0)

        with col2:
            session_duration = st.number_input("Durée de la session en secondes (session_duration)",
                                                  min_value=0.0, value=120.0)
            ip_reputation_score = st.slider("Score de réputation IP (ip_reputation_score)",
                                              min_value=0.0, max_value=1.0, value=0.5)
            unusual_time_access = st.selectbox("Accès à une heure inhabituelle (unusual_time_access)",
                                                 options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

        with col3:
            protocol_type = st.selectbox("Type de protocole (protocol_type)", options=protocols)
            encryption_used = st.selectbox("Type de chiffrement (encryption_used)", options=encryptions)
            browser_type = st.selectbox("Type de navigateur (browser_type)", options=browsers)

        submitted = st.form_submit_button("🚀 Prédire", type="primary")

    if submitted:
        manual_df = pd.DataFrame([{
            "session_id": "manual_session",
            "network_packet_size": network_packet_size,
            "protocol_type": protocol_type,
            "login_attempts": login_attempts,
            "session_duration": session_duration,
            "encryption_used": encryption_used,
            "ip_reputation_score": ip_reputation_score,
            "failed_logins": failed_logins,
            "browser_type": browser_type,
            "unusual_time_access": unusual_time_access,
        }])

        X_manual, _ = preprocess(manual_df, scaler, feature_cols, drop_na=False)

        prediction = model.predict(X_manual)[0]
        probability = model.predict_proba(X_manual)[0, 1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **Attaque détectée** — probabilité : {probability:.2%}")
        else:
            st.success(f"✅ **Session normale** — probabilité d'attaque : {probability:.2%}")

        st.subheader("🔬 Pourquoi cette prédiction ?")
        with st.spinner("Calcul de l'explication SHAP..."):
            shap_values_manual_raw = explainer.shap_values(X_manual)
            sv_2d, base_value = extract_shap_for_positive_class(shap_values_manual_raw, explainer.expected_value)
            sv = sv_2d[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            base_value, sv, feature_names=X_manual.columns.tolist(), show=False
        )
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)

        st.info(
            "**Lecture du graphique** : les barres rouges augmentent la probabilité "
            "d'attaque, les barres bleues la diminuent. La valeur de base (E[f(x)]) "
            "représente la prédiction moyenne sur l'ensemble du dataset."
        )
