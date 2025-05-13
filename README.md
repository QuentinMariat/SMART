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

## Application web

L'application propose une interface web permettant de charger un lien YouTube et d'afficher les résultats de l'analyse de sentiments dans les commentaires, puis de l'afficher.

### 🚀 Lancer l'application

Dans un premier temps, il faut ajouter le fichier avec les clefs d'API à l'adresse suivante : src\WebApp\backend\scrapping\config.json

Il faut ensuite se placer à l'emplacement des fichiers du backend
```bash
cd cd src/WebApp/backend
```
Il faut exectuter la commande suivante :
```bash
uvicorn main:app --reload
```

Les différentes endpointes sont les suivants :
| Méthode | Endpoint           | Description                                      |
| ------- | ----------         | ------------------------------------------------ |
| `POST`  | `/analyze/twitter` | Analyse des réponses d'un tweet                  |
| `POST`  | `/analyze/youtube` | Analyse des commentaires d'une vidéo Youtube     |
| `GET`   | `/docs`            | Documentation interactive de l'API (Swagger UI)  |


Par défaut, la base de l'url est localhost:8000. Si le backend est executé ailleurs qu'en locale, il faut remplacer localhost:8000 par l'adresse du serveur.

Il suffit ensuite d'executer le fichier src\WebApp\frontend\index.html pour acceder à la page web.


