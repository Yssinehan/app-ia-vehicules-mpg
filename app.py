"""Tableau de bord Streamlit d'analyse et de prédiction de consommation automobile."""

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


URL_DONNEES = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
COLONNES_REQUISES = [
    "mpg",
    "cylinders",
    "horsepower",
    "weight",
    "model_year",
    "origin",
]
COLONNES_NUMERIQUES = ["cylinders", "horsepower", "weight", "model_year"]
COLONNES_MODELE = COLONNES_NUMERIQUES + ["origin"]

st.set_page_config(
    page_title="Analyse automobile — MPG",
    page_icon="🚗",
    layout="wide",
)


@st.cache_data(show_spinner="Chargement des données…")
def charger_donnees() -> pd.DataFrame:
    """Télécharge le CSV et vérifie la présence des colonnes utilisées."""
    donnees = pd.read_csv(URL_DONNEES)
    colonnes_absentes = sorted(set(COLONNES_REQUISES) - set(donnees.columns))
    if colonnes_absentes:
        raise ValueError(
            "Le fichier téléchargé ne contient pas les colonnes attendues : "
            + ", ".join(colonnes_absentes)
        )
    return donnees


@st.cache_resource(show_spinner="Entraînement du modèle…")
def entrainer_modele(donnees: pd.DataFrame) -> Pipeline:
    """Entraîne le modèle sur les lignes complètes pour ses variables d'entrée."""
    donnees_modele = donnees[COLONNES_REQUISES].dropna()
    if donnees_modele.empty:
        raise ValueError("Aucune ligne complète n'est disponible pour entraîner le modèle.")

    preparation = ColumnTransformer(
        transformers=[
            ("origine", OneHotEncoder(handle_unknown="ignore"), ["origin"]),
        ],
        remainder="passthrough",
    )
    modele = Pipeline(
        steps=[
            ("preparation", preparation),
            (
                "regression",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    modele.fit(donnees_modele[COLONNES_MODELE], donnees_modele["mpg"])
    return modele


st.title("🚗 Analyse et prédiction de la consommation automobile")
st.caption(
    "Analyse descriptive et estimation du rendement énergétique à partir du jeu "
    "de données Auto MPG. La cible `mpg` est exprimée en miles par gallon américain."
)

try:
    df = charger_donnees()
except (ValueError, OSError, pd.errors.ParserError) as erreur:
    st.error(f"Impossible de charger ou de vérifier le jeu de données : {erreur}")
    st.stop()

onglet_analyse, onglet_prediction = st.tabs(
    ["📊 Analyse descriptive", "🤖 Prédiction"]
)

with onglet_analyse:
    st.subheader("Explorer les véhicules")

    with st.sidebar:
        st.header("Filtres de l'analyse")
        origines_disponibles = sorted(df["origin"].dropna().unique().tolist())
        origines_choisies = st.multiselect(
            "Origine",
            options=origines_disponibles,
            default=origines_disponibles,
            format_func=lambda origine: str(origine).capitalize(),
        )

        annees = sorted(df["model_year"].dropna().astype(int).unique().tolist())
        plage_annees = st.slider(
            "Année modèle",
            min_value=1900 + annees[0],
            max_value=1900 + annees[-1],
            value=(1900 + annees[0], 1900 + annees[-1]),
            step=1,
        )

        cylindres_disponibles = sorted(
            df["cylinders"].dropna().astype(int).unique().tolist()
        )
        cylindres_choisis = st.multiselect(
            "Nombre de cylindres",
            options=cylindres_disponibles,
            default=cylindres_disponibles,
        )

    masque = (
        df["origin"].isin(origines_choisies)
        & df["model_year"].between(plage_annees[0] - 1900, plage_annees[1] - 1900)
        & df["cylinders"].isin(cylindres_choisis)
    )
    donnees_filtrees = df.loc[masque].copy()

    if donnees_filtrees.empty:
        st.info("Aucun véhicule ne correspond aux filtres sélectionnés.")
    else:
        puissance_moyenne = donnees_filtrees["horsepower"].mean()
        consommation_moyenne = donnees_filtrees["mpg"].mean()
        indicateur_1, indicateur_2, indicateur_3 = st.columns(3)
        indicateur_1.metric("Véhicules filtrés", f"{len(donnees_filtrees):,}".replace(",", " "))
        indicateur_2.metric(
            "Puissance moyenne",
            f"{puissance_moyenne:.1f} ch" if pd.notna(puissance_moyenne) else "Indisponible",
        )
        indicateur_3.metric(
            "Consommation moyenne",
            f"{consommation_moyenne:.1f} mpg"
            if pd.notna(consommation_moyenne)
            else "Indisponible",
        )

        gauche, droite = st.columns(2)
        with gauche:
            st.markdown("#### Poids et consommation")
            st.scatter_chart(
                donnees_filtrees,
                x="weight",
                y="mpg",
                color="origin",
                x_label="Poids (livres)",
                y_label="Consommation (mpg)",
                width="stretch",
            )

        with droite:
            st.markdown("#### Consommation selon l'origine")
            comparaison_origines = (
                donnees_filtrees.groupby("origin", as_index=False)["mpg"]
                .mean()
                .rename(columns={"mpg": "Consommation moyenne (mpg)"})
            )
            st.bar_chart(
                comparaison_origines,
                x="origin",
                y="Consommation moyenne (mpg)",
                x_label="Origine",
                y_label="Consommation moyenne (mpg)",
                width="stretch",
            )

        with st.expander("Afficher les données filtrées"):
            st.dataframe(donnees_filtrees, width="stretch", hide_index=True)

with onglet_prediction:
    st.subheader("Estimer la consommation d'un véhicule")
    st.write(
        "Renseignez les caractéristiques ci-dessous. Le modèle Random Forest "
        "est entraîné sur les observations complètes du jeu de données."
    )

    try:
        modele = entrainer_modele(df)
    except (ValueError, KeyError) as erreur:
        st.error(f"Impossible d'entraîner le modèle : {erreur}")
        st.stop()

    donnees_entrainement = df[COLONNES_REQUISES].dropna()
    colonnes_saisie = st.columns(2)
    with colonnes_saisie[0]:
        cylindres = st.selectbox(
            "Nombre de cylindres",
            options=sorted(donnees_entrainement["cylinders"].astype(int).unique()),
            index=0,
        )
        puissance = st.slider(
            "Puissance (horsepower)",
            min_value=int(donnees_entrainement["horsepower"].min()),
            max_value=int(donnees_entrainement["horsepower"].max()),
            value=int(donnees_entrainement["horsepower"].median()),
        )
        poids = st.slider(
            "Poids (weight, en livres)",
            min_value=int(donnees_entrainement["weight"].min()),
            max_value=int(donnees_entrainement["weight"].max()),
            value=int(donnees_entrainement["weight"].median()),
            step=10,
        )
    with colonnes_saisie[1]:
        annee_affichee = st.slider(
            "Année modèle",
            min_value=1900 + int(donnees_entrainement["model_year"].min()),
            max_value=1900 + int(donnees_entrainement["model_year"].max()),
            value=1900 + int(donnees_entrainement["model_year"].median()),
            step=1,
        )
        origine = st.selectbox(
            "Origine",
            options=sorted(donnees_entrainement["origin"].unique()),
            format_func=lambda valeur: str(valeur).capitalize(),
        )

    vehicule = pd.DataFrame(
        [
            {
                "cylinders": cylindres,
                "horsepower": puissance,
                "weight": poids,
                "model_year": annee_affichee - 1900,
                "origin": origine,
            }
        ],
        columns=COLONNES_MODELE,
    )
    prediction = float(modele.predict(vehicule)[0])
    st.metric("Consommation estimée", f"{prediction:.1f} mpg")
    st.caption(
        "Cette estimation est indicative : elle dépend des véhicules présents "
        "dans le jeu de données et n'est pas une mesure réelle."
    )
