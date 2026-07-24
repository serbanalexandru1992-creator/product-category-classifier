"""
train_model.py
---------------
Antreneaza un model de clasificare care prezice categoria unui produs
(coloana "Category Label") pe baza titlului produsului (coloana
"Product Title"), plus cateva caracteristici suplimentare extrase din titlu.

Ce face acest script, pas cu pas:
1. Incarca products.csv
2. Curata datele (elimina randuri incomplete)
3. Adauga caracteristici suplimentare din titlu (vezi features.py)
4. Imparte datele in antrenament/testare
5. Antreneaza si compara mai multe modele de clasificare
6. Alege cel mai bun model (dupa F1 - weighted, potrivit pt. multe categorii)
7. Salveaza modelul castigator (impreuna cu toata pregatirea datelor,
   ca un singur "pipeline") in fisierul category_model.pkl

Ruleaza acest script din terminal cu:
    python3 train_model.py
"""

import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from features import build_features, NUMERIC_FEATURE_COLUMNS

TITLE_COLUMN = "Product Title"
CATEGORY_COLUMN = "Category Label"
DATA_PATH = "products.csv"
MODEL_OUTPUT_PATH = "category_model.pkl"


# ---------------------------------------------------------------------------
# PASUL 1: Incarcarea datelor
# ---------------------------------------------------------------------------
print("Incarc datele din", DATA_PATH, "...")
df = pd.read_csv(DATA_PATH)

# IMPORTANT: fisierul products.csv real are spatii ascunse in unele nume de
# coloane (ex: " Category Label" in loc de "Category Label"). Curatam
# numele coloanelor imediat dupa incarcare, ca sa nu conteze exact cum
# sunt scrise in fisierul original.
df.columns = df.columns.str.strip()

print(f"Numar total de produse incarcate: {len(df)}")
print("\nValori lipsa pe coloane relevante:")
print(df[[TITLE_COLUMN, CATEGORY_COLUMN]].isnull().sum())


# ---------------------------------------------------------------------------
# PASUL 2: Curatarea datelor
# ---------------------------------------------------------------------------
# Eliminam randurile fara titlu sau fara categorie - nu ne ajuta la antrenare.
df = df.dropna(subset=[TITLE_COLUMN, CATEGORY_COLUMN])

# Standardizam categoria (spatii in plus, litere mari/mici pot crea
# "categorii duplicate" artificiale, ex: "Laptops" vs "laptops ").
df[CATEGORY_COLUMN] = df[CATEGORY_COLUMN].astype(str).str.strip()

# PROBLEMA GASITA IN DATELE REALE: aceeasi categorie apare scrisa in mai
# multe feluri - ex: "CPU" vs "CPUs", "Mobile Phone" vs "Mobile Phones",
# "fridge" (litere mici) vs "Fridges". Fara aceasta corectie, modelul le-ar
# trata ca fiind categorii complet diferite, ceea ce ii scade artificial
# performanta pe acele categorii (foarte putine exemple per varianta).
#
# Solutie simpla: normalizam fiecare categorie (litere mici + eliminam un
# eventual "s" de plural de la final), grupam variantele care ajung la
# aceeasi forma normalizata, si le inlocuim pe toate cu forma cea mai
# frecventa din acel grup (ex: "CPU" si "CPUs" devin ambele "CPUs").
def _normalize_category(cat: str) -> str:
    cat = cat.strip().lower()
    if cat.endswith("s") and not cat.endswith("ss"):
        cat = cat[:-1]
    return cat

df["_category_normalized"] = df[CATEGORY_COLUMN].apply(_normalize_category)
canonical_names = (
    df.groupby("_category_normalized")[CATEGORY_COLUMN]
    .agg(lambda values: values.value_counts().idxmax())
)
df[CATEGORY_COLUMN] = df["_category_normalized"].map(canonical_names)
df = df.drop(columns=["_category_normalized"])

# Eliminam categoriile cu foarte putine exemple (sub 2 produse) - acestea
# nu pot fi impartite corect intre antrenament si testare cu "stratify".
category_counts = df[CATEGORY_COLUMN].value_counts()
valid_categories = category_counts[category_counts >= 2].index
df = df[df[CATEGORY_COLUMN].isin(valid_categories)]

print(f"\nNumar de produse ramase dupa curatare: {len(df)}")
print(f"Numar de categorii distincte: {df[CATEGORY_COLUMN].nunique()}")


# ---------------------------------------------------------------------------
# PASUL 3: Ingineria caracteristicilor (feature engineering)
# ---------------------------------------------------------------------------
# Adaugam caracteristici suplimentare extrase din titlu (numar de cuvinte,
# prezenta cifrelor, cuvinte scrise cu majuscule etc.) - vezi features.py
# pentru explicatii complete despre fiecare caracteristica.
df = build_features(df, title_column=TITLE_COLUMN)


# ---------------------------------------------------------------------------
# PASUL 4: Impartirea in X / y si train/test split
# ---------------------------------------------------------------------------
feature_columns = [TITLE_COLUMN] + NUMERIC_FEATURE_COLUMNS
X = df[feature_columns]
y = df[CATEGORY_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nProduse de antrenament: {len(X_train)}")
print(f"Produse de testare: {len(X_test)}")


# ---------------------------------------------------------------------------
# PASUL 5: Preprocesare - combinam text (TF-IDF) cu caracteristici numerice
# ---------------------------------------------------------------------------
# ColumnTransformer aplica transformari diferite pe coloane diferite:
# - titlul produsului trece prin TfidfVectorizer (text -> numere)
# - caracteristicile numerice sunt scalate cu MinMaxScaler, care le aduce
#   in intervalul [0, 1]. Am ales MinMaxScaler (in loc de StandardScaler)
#   in mod deliberat: StandardScaler poate produce valori negative, iar
#   MultinomialNB (unul din modelele comparate mai jos) accepta DOAR
#   valori numerice nenegative - altfel antrenarea acelui model crapa.
preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(max_features=5000), TITLE_COLUMN),
        ("numeric", MinMaxScaler(), NUMERIC_FEATURE_COLUMNS),
    ]
)


# ---------------------------------------------------------------------------
# PASUL 6: Antrenarea si compararea mai multor modele
# ---------------------------------------------------------------------------
# Fiecare model este impachetat intr-un Pipeline impreuna cu preprocesarea,
# astfel incat sa putem salva un singur obiect care stie cum sa transforme
# date noi (brute) in predictii finale.
candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVC": LinearSVC(),
}

best_model_name = None
best_pipeline = None
best_f1 = -1.0

for name, classifier in candidate_models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("=" * 70)
    print(f"Model: {name}")
    print(f"Acuratete: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print("\nRaport de clasificare (rezumat):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Pastram modelul cu cel mai bun F1 (weighted) - alegem F1 in loc de
    # acuratete simpla pentru ca avem multe categorii, posibil dezechilibrate;
    # F1 (weighted) reflecta mai corect performanta pe toate categoriile.
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_pipeline = pipeline

print("\n" + "=" * 70)
print(f"Cel mai bun model: {best_model_name} (F1 weighted = {best_f1:.4f})")


# ---------------------------------------------------------------------------
# PASUL 7: Salvarea modelului castigator
# ---------------------------------------------------------------------------
# Salvam intregul pipeline (preprocesare + model), nu doar clasificatorul -
# astfel, la predictie, e suficient sa incarcam acest fisier si sa ii dam
# date brute (titlul produsului), fara sa reconstruim manual vectorizarea.
with open(MODEL_OUTPUT_PATH, "wb") as f:
    pickle.dump(best_pipeline, f)

print(f"\nModelul a fost salvat in fisierul: {MODEL_OUTPUT_PATH}")