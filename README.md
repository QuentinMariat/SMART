# SMART

## Installation

### Création de venv et installations des requirements de base (utiliser python 3.12.X):

```bash
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```

### Si vous voulez utiliser CUDA pour aller plus vite (carte graphique)

Check quelle version de pytorch est installée :

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Si version **2.7.0+cpu** : désinstaller et réinstaller version cuda :

```bash
pip uninstall torch
pip3 install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

## Execution

### lancer l'entrainement :

```bash
python3 -m src.train
```

### lancer l'évaluation du modèle :

```bash
 python3 -m src.evaluate
```

### scrapper les commentaires d'une page youtube (attention : clé API nécessaire) :

```bash
python3 src/youtube_scrapper.py "https://www.youtube.com/watch?v=LIEN_VIDEO" "API_KEY"
```

### prédire les sentiments globaux :

```bash
PYTHONPATH=. python3 src/predict.py src/scrapping/output/youtube_comments_LIEN_VIDEO.csv
```
