"""
features.py
------------
Functii de inginerie a caracteristicilor (feature engineering), folosite
IDENTIC atat la antrenarea modelului (train_model.py / notebook), cat si
la predictie (predict_category.py).

De ce un fisier separat?
Pentru ca modelul a fost antrenat folosind aceste caracteristici suplimentare
(nu doar textul titlului), trebuie sa calculam EXACT aceleasi caracteristici
si atunci cand primim un titlu nou de la utilizator. Daca am scrie codul de
doua ori (o data la antrenare, o data la predictie) si am uita sa il
sincronizam, modelul ar primi date "diferite" fata de ce a invatat, si ar
da predictii gresite. Punand codul intr-un singur loc, eliminam acest risc.
"""

import re
import pandas as pd


def build_features(df: pd.DataFrame, title_column: str = "Product Title") -> pd.DataFrame:
    """
    Primeste un DataFrame cu o coloana de titluri de produse si adauga
    caracteristici numerice suplimentare, utile pentru clasificare:

    - title_word_count       : numarul de cuvinte din titlu
    - title_char_count       : numarul de caractere din titlu
    - has_digit               : 1 daca titlul contine cel putin o cifra, altfel 0
    - has_special_char        : 1 daca titlul contine caractere speciale (non alfanumerice, in afara de spatiu)
    - has_all_caps_word       : 1 daca exista un cuvant scris integral cu majuscule (ex: "USB", "LED")
    - longest_word_length     : lungimea celui mai lung cuvant din titlu

    Returneaza acelasi DataFrame, cu coloanele noi adaugate.
    """
    df = df.copy()

    # Ne asiguram ca titlul e text (string), nu NaN sau alt tip.
    titles = df[title_column].astype(str)

    df["title_word_count"] = titles.apply(lambda t: len(t.split()))
    df["title_char_count"] = titles.apply(len)
    df["has_digit"] = titles.apply(lambda t: int(any(ch.isdigit() for ch in t)))
    df["has_special_char"] = titles.apply(
        lambda t: int(bool(re.search(r"[^A-Za-z0-9\s]", t)))
    )
    df["has_all_caps_word"] = titles.apply(
        lambda t: int(any(w.isupper() and len(w) > 1 for w in t.split()))
    )
    df["longest_word_length"] = titles.apply(
        lambda t: max((len(w) for w in t.split()), default=0)
    )

    return df


# Numele coloanelor numerice generate mai sus - le folosim si in train_model.py
# si in predict_category.py, ca sa nu le rescriem manual de fiecare data.
NUMERIC_FEATURE_COLUMNS = [
    "title_word_count",
    "title_char_count",
    "has_digit",
    "has_special_char",
    "has_all_caps_word",
    "longest_word_length",
]
