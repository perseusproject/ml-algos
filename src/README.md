# TP — Algorithms for Data Analysis (CSC 43042 EP, S. Oudot)

Travaux pratiques Jupyter qui font coder **chaque méthode du cours** à partir de NumPy, la comparer à ses voisines du même chapitre, et l'appliquer à un problème de **recherche quantitative**.

## Contenu

```
TP_DataAnalysis/
├── README.md              ← ce fichier
├── tp_utils.py            ← style des graphiques, check() / observe() pour les tests
├── tp_data.py             ← données réelles (téléchargement + cache) ou simulées
├── data/                  ← cache CSV, créé au premier téléchargement
├── exercices/             ← énoncés : blocs « # TODO » à compléter
└── corriges/              ← corrigés exécutés de bout en bout (avec sorties)
```

| # | Notebook | Slides | Méthodes codées | Application quant |
|---|---|---|---|---|
| 01 | `01_nn_search` | 19–40 | linear scan, k-d tree (construction, defeatist, backtracking, k-NN), malédiction de la dimension | recherche d'analogues de marché (ETF sectoriels) |
| 02 | `02_kmeans` | 41–63 | Lloyd, 4 initialisations dont k-means++, DP exacte 1D, coude, silhouette | régimes de volatilité du S&P 500 ; clusters d'actions vs secteurs GICS |
| 03 | `03_hierarchical` | 65–88 | union-find, single-linkage (Kruskal), AHC (single/complete/average/Ward), ultramétrique, stabilité, UPGMA | dendrogramme du marché ; **Hierarchical Risk Parity** + backtest |
| 04 | `04_density` | 89–117 | gaussienne (MLE), **EM** pour GMM, BIC, histogramme, KDE (3 noyaux), Silverman, CV | queues du S&P 500 ; régimes par GMM ; **VaR 99 %** et backtest |
| 05 | `05_knn_cv` | 118–135 | k-NN (régression, classification), classifieur de Bayes, holdout / r-fold / LOO / Monte-Carlo, matrice de confusion, précision/rappel/F1, ROC/AUC | prévision de semaines agitées ; **fuite d'information** de la CV sur séries temporelles |
| 06 | `06_linear_regression` | 137–157 | OLS (équations normales, QR), R², erreurs standard, Gauss-Markov, ridge (+ LOO gratuite), lasso (descente par coordonnées), fonctions de base, kernel ridge | modèle **Fama-French** ; combiner 60 signaux faibles (ridge vs lasso) |
| 07 | `07_linear_classification` | 159–189 | régression sur indicatrices (masking), logistique par **Newton**, multi-classe, **SVM dual**, SVM par sous-gradient, noyaux, logistique à noyau | **déséquilibre du carnet d'ordres** → prochain mouvement ; calibration |
| 08 | `08_ensembles` | 191–214 | CART (régression, Gini), forêt aléatoire, OOB, MDI, MDA, souches pondérées, **AdaBoost** + borne | prédiction de rendements en coupe transversale ; piège de la MDI |
| 09 | `09_neural_networks` | 215–242 | perceptron (borne de Novikoff), **MLP + rétropropagation**, contrôle de gradient, arrêt précoce / weight decay / dropout, **convolution** et couche locale (expérience de LeCun) | prévision de volatilité : **HAR** vs MLP |
| 10 | `10_features_pca` | 243–270 | sac de mots, n-grammes, tf-idf, graphlets, histogrammes d'orientations (SIFT), spin images, plongement retardé, **PCA par SVD**, quantification vectorielle | sentiment de titres financiers ; réseau de corrélations ; **Marchenko-Pastur** ; courbe des taux (niveau/pente/courbure) ; motifs de prix |
| 11 | `11_synthese` | — | tableaux « quel algo, quand », pièges de la recherche quant, auto-évaluation (15 questions) | — |

Durée indicative : 2 à 4 h par notebook. Ordre conseillé : 1 → 10 (les TP 5 à 9 réutilisent les notions de validation du TP 5), puis la synthèse.

## Installation

Python ≥ 3.10.

```bash
pip install numpy scipy pandas matplotlib scikit-learn jupyterlab yfinance
jupyter lab
```

`yfinance` est optionnel : sans lui (ou sans internet), tout fonctionne avec des données simulées.

## Méthode de travail

1. Ouvrez `exercices/XX_….ipynb` et exécutez les cellules dans l'ordre.
2. Complétez chaque bloc `# TODO` (les indices sont en commentaire), puis exécutez la cellule **Test** qui suit : `✅` = correct, `❌` = erreur explicite.
3. Les cellules `observe(...)` affichent `✅` ou `⚠️` sans bloquer : ce sont des **constats empiriques** qui peuvent varier selon les données — un `⚠️` est une invitation à discuter, pas une erreur de code.
4. Répondez aux **Questions** (cellules « ✍️ Votre réponse »), puis comparez avec `corriges/XX_…_corrige.ipynb`.
5. Chaque notebook se termine par un tableau de synthèse « quel algorithme, quand ».

Toutes les implémentations sont validées contre une référence (scikit-learn, SciPy ou force brute) : vous saurez que votre code est juste, pas seulement qu'il « tourne ».

## Données

Le module `tp_data.py` suit toujours la même logique : **cache local** (`data/*.csv`) → **téléchargement** → **simulation** si le téléchargement échoue (le notebook l'indique : `[tp_data] … données SIMULÉES`).

| Source | Contenu | Notebooks | Accès |
|---|---|---|---|
| Yahoo Finance (`yfinance`) | ~64 grandes capitalisations US (6–7 par secteur GICS), 2015–2024 | 2, 3, 6, 10 | gratuit, sans clé |
| Yahoo Finance (`yfinance`) | SPY + 9 ETF sectoriels SPDR, 2005–2024 | 1 | gratuit, sans clé |
| Yahoo Finance (`yfinance`) | SPY quotidien, 2000–2024 | 2, 4, 5, 9, 10 | gratuit, sans clé |
| [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | facteurs Fama-French 3 quotidiens | 6 | téléchargement direct (zip) |
| [FRED](https://fred.stlouisfed.org/) | taux du Trésor US, 10 maturités (DGS3MO … DGS30) | 10 | téléchargement direct (CSV) |
| scikit-learn | chiffres manuscrits 8×8 (`load_digits`) | 9, 10 | inclus, hors ligne |
| simulé | mélange de gaussiennes du cours (erreur de Bayes ≈ 21 %), carnet d'ordres, panel d'actions, titres d'actualité, signaux faibles | 2–10 | hors ligne, vérité connue |

- Pour **forcer la simulation** (hors ligne, résultats reproductibles) : `tp_data.USE_REAL_DATA = False` en début de notebook.
- Pour **retélécharger** : supprimez le fichier correspondant dans `data/`.
- Les simulations reproduisent les faits stylisés utiles : facteur marché + facteurs sectoriels, volatilité GARCH, queues épaisses (Student-t), courbe des taux Nelson-Siegel à 3 facteurs. Quand la vérité est connue (bêtas Fama-French simulés, mélange du cours, carnet d'ordres), les notebooks vérifient que les méthodes la retrouvent.
- Pour le TP 10, le jeu réel *Financial PhraseBank* (Malo et al., 2014 ; Hugging Face `takala/financial_phrasebank`) peut remplacer le générateur de titres sans changer le code d'extraction.

**Important :** les corrigés fournis ont été exécutés avec les **données simulées** (l'environnement de génération n'avait pas accès à Yahoo, FRED ni à la bibliothèque de K. French). Avec les données réelles, les chiffres changeront : c'est voulu, et les questions de discussion portent sur ces différences (par exemple, la neutralisation du facteur marché améliore nettement le clustering sectoriel sur données réelles).

## Références

- S. Oudot, *Algorithms for Data Analysis in Python*, CSC 43042 EP, École polytechnique (slides du cours).
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning* (jeu du mélange de gaussiennes, §2.3).
- López de Prado, *Building Diversified Portfolios that Outperform Out of Sample* (HRP), 2016 ; *Advances in Financial Machine Learning* (fuite d'information, MDI/MDA), 2018.
- Corsi, *A Simple Approximate Long-Memory Model of Realized Volatility* (HAR), 2009.
- Gu, Kelly, Xiu, *Empirical Asset Pricing via Machine Learning*, 2020.
- Bouchaud, Potters, *Theory of Financial Risk and Derivative Pricing* (matrices aléatoires), 2003.
- Litterman, Scheinkman, *Common Factors Affecting Bond Returns*, 1991.
