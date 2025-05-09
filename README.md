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
python3 src/youtube_scrapper.py "https://www.youtube.com/watch?v=LIEN_VIDEO" "API_KEY"
```

prédire les sentiments globaux :

```bash
PYTHONPATH=. python3 src/predict.py src/scrapping/output/youtube_comments_LIEN_VIDEO.csv
```
