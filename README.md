# Predicția categoriei produsului pe baza titlului

Proiect ML care sugerează automat categoria unui produs (ex: "Mobile Phones",
"Laptops", "Fridge Freezers") pe baza titlului său, folosind un set de date
real cu peste 30.000 de produse.

## Structura proiectului

```
.
├── products.csv                       # setul de date (produse + categorii)
├── features.py                        # inginerie caracteristici (partajat)
├── train_model.py                     # antreneaza si salveaza modelul
├── predict_category.py                # testare interactiva a modelului
├── product_category_analysis.ipynb    # analiza completa, pas cu pas
├── category_model.pkl                 # modelul antrenat (generat de train_model.py)
└── README.md
```

## Cerințe

- Python 3.10+
- Biblioteci: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

Instalare:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Cum rulezi proiectul

### 1. Antrenarea modelului

```bash
python3 train_model.py
```

Acest script:
- încarcă și curăță `products.csv`;
- adaugă caracteristici suplimentare din titlu (număr de cuvinte, prezența
  cifrelor, cuvinte scrise cu majuscule etc.);
- antrenează și compară trei modele (Logistic Regression, Multinomial Naive
  Bayes, Linear SVC);
- salvează modelul cu cea mai bună performanță (F1 weighted) în
  `category_model.pkl`.

### 2. Testarea interactivă

După ce `category_model.pkl` există (adică ai rulat pasul 1 măcar o dată):

```bash
python3 predict_category.py
```

Introdu titlul unui produs (de exemplu `iphone 7 32gb gold`) și scriptul
afișează categoria prezisă. Scrie `exit` ca să închei.

### 3. Analiza completă (notebook)

Deschide `product_category_analysis.ipynb` în Jupyter sau VS Code, pentru
explorarea datelor, ingineria caracteristicilor, compararea modelelor și
vizualizări (distribuția categoriilor, matricea de confuzie).

## Despre setul de date

`products.csv` conține următoarele coloane relevante pentru acest proiect:

- `Product Title` – titlul produsului (folosit ca intrare pentru model)
- `Category Label` – categoria țintă (ce încearcă modelul să prezică)

Alte coloane disponibile în set (Merchant ID, Number of Views, Merchant
Rating, Listing Date) nu sunt folosite în versiunea curentă a modelului, dar
pot fi explorate ca posibile caracteristici suplimentare într-o versiune
viitoare.

## Observații și posibile îmbunătățiri

*(Secțiune de completat pe baza propriei experiențe de dezvoltare – ce ai
încercat, ce a funcționat, ce ai îmbunătăți într-o versiune viitoare.)*
