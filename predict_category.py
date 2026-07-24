"""
predict_category.py
--------------------
Incarca modelul deja antrenat (category_model.pkl) si permite testarea
interactiva: utilizatorul introduce titlul unui produs, iar modelul
sugereaza categoria.

IMPORTANT: trebuie sa rulezi mai intai train_model.py o data, ca sa
existe fisierul category_model.pkl. Acest script doar il incarca si il
foloseste - nu antreneaza nimic.

Ruleaza acest script din terminal cu:
    python3 predict_category.py
"""

import pickle

import pandas as pd

from features import build_features

MODEL_PATH = "category_model.pkl"
TITLE_COLUMN = "Product Title"


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_title(pipeline, title: str) -> str:
    """
    Primeste un singur titlu (text simplu) si returneaza categoria prezisa.

    Modelul a fost antrenat pe un DataFrame cu coloana "Product Title" plus
    caracteristici suplimentare (numar de cuvinte, prezenta cifrelor etc.),
    asa ca reconstruim aceeasi structura pentru un singur titlu nou, folosind
    functia build_features din features.py - EXACT aceeasi functie folosita
    si la antrenare.
    """
    single_row_df = pd.DataFrame({TITLE_COLUMN: [title]})
    single_row_df = build_features(single_row_df, title_column=TITLE_COLUMN)

    prediction = pipeline.predict(single_row_df)
    return prediction[0]


if __name__ == "__main__":
    print("Incarc modelul din", MODEL_PATH, "...")
    model_pipeline = load_model(MODEL_PATH)
    print("Model incarcat cu succes.\n")

    print("--- Testare interactiva ---")
    print("Introdu titlul unui produs pentru a afla categoria prezisa.")
    print("Scrie 'exit' pentru a iesi.\n")

    while True:
        user_title = input("Titlu produs: ")

        if user_title.strip().lower() == "exit":
            print("La revedere!")
            break

        if not user_title.strip():
            print(">> Te rog introdu un titlu (nu poate fi gol).\n")
            continue

        predicted_category = predict_title(model_pipeline, user_title)
        print(f">> Categorie prezisa: {predicted_category}\n")
