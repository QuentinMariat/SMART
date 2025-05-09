# SMART

lancer l'entrainement :

```bash
python3 -m src.train
```

lancer l'évaluation du modèle :

```bash
 python3 -m src.evaluate
```

scrapper les commentaires d'une page youtube (attention : clé API nécessaire) :

```bash
python3 src/youtube_scrapper.py "https://www.youtube.com/watch?v=LIEN"
```

exemple :

```bash
python3 src/youtube_scrapper.py "https://www.youtube.com/watch?v=CMDPlirF4tg"
```

prédire les sentiments globaux :

```bash
PYTHONPATH=. python3 src/predict.py src/scrapping/output/youtube_comments_LIEN.csv
```

exemple :

```bash
PYTHONPATH=. python3 src/predict.py src/scrapping/output/youtube_comments_CMDPlirF4tg.csv
```
