#!/usr/bin/env python3
"""
Script de nettoyage du cache pour supprimer les anciens fichiers CSV d'analyse.
"""
import os
import sys
import logging
import glob
import argparse
from datetime import datetime, timedelta

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def clean_cache(days=7, force=False, keep_latest=True):
    """
    Nettoie les fichiers CSV plus anciens que le nombre de jours spécifié.
    
    Args:
        days (int): Âge maximum des fichiers en jours
        force (bool): Si True, supprime les fichiers sans confirmation
        keep_latest (bool): Conserver le dernier fichier pour chaque ID de vidéo
    """
    # Obtenir le chemin du répertoire de sortie
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    
    if not os.path.exists(output_dir):
        logger.warning(f"Le répertoire output n'existe pas: {output_dir}")
        return
    
    # Obtenir tous les fichiers CSV
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    
    if not csv_files:
        logger.info("Aucun fichier CSV trouvé")
        return
    
    # Calculer la date limite
    now = datetime.now()
    cutoff_date = now - timedelta(days=days)
    
    # Organiser les fichiers par ID de vidéo
    files_by_video_id = {}
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        
        # Extraire l'ID de la vidéo
        video_id = None
        if file_name.startswith("youtube_comments_"):
            # Format: youtube_comments_VIDEO_ID.csv ou youtube_comments_VIDEO_ID_labeled_*.csv
            if "_labeled_" in file_name:
                parts = file_name.split("_labeled_")[0].split("youtube_comments_")
                if len(parts) > 1:
                    video_id = parts[1]
            else:
                parts = file_name.split(".")[0].split("youtube_comments_")
                if len(parts) > 1:
                    video_id = parts[1]
        
        if video_id:
            if video_id not in files_by_video_id:
                files_by_video_id[video_id] = []
            
            file_stat = os.stat(file_path)
            file_date = datetime.fromtimestamp(file_stat.st_mtime)
            
            files_by_video_id[video_id].append({
                "path": file_path,
                "name": file_name,
                "date": file_date,
                "size": file_stat.st_size
            })
    
    # Trier les fichiers par date pour chaque ID de vidéo
    files_to_delete = []
    kept_files = []
    
    for video_id, files in files_by_video_id.items():
        # Trier les fichiers par date (le plus récent en premier)
        sorted_files = sorted(files, key=lambda x: x["date"], reverse=True)
        
        # Conserver le fichier le plus récent si demandé
        if keep_latest and sorted_files:
            kept_files.append(sorted_files[0])
            sorted_files = sorted_files[1:]
        
        # Marquer les fichiers anciens pour suppression
        for file_info in sorted_files:
            if file_info["date"] < cutoff_date:
                files_to_delete.append(file_info)
    
    # Afficher un résumé
    logger.info(f"Total des fichiers CSV: {len(csv_files)}")
    logger.info(f"Fichiers à supprimer: {len(files_to_delete)}")
    logger.info(f"Fichiers conservés: {len(kept_files)}")
    
    # Suppression interactive ou forcée
    if files_to_delete:
        if force:
            for file_info in files_to_delete:
                try:
                    os.remove(file_info["path"])
                    logger.info(f"Supprimé: {file_info['name']} ({file_info['date'].strftime('%Y-%m-%d %H:%M:%S')})")
                except Exception as e:
                    logger.error(f"Erreur lors de la suppression de {file_info['path']}: {str(e)}")
            
            logger.info(f"{len(files_to_delete)} fichiers supprimés avec succès")
        else:
            logger.info("Fichiers à supprimer:")
            for i, file_info in enumerate(files_to_delete, 1):
                logger.info(f"{i}. {file_info['name']} - {file_info['date'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            confirmation = input(f"Supprimer {len(files_to_delete)} fichiers? (o/n): ")
            if confirmation.lower() in ["o", "oui", "y", "yes"]:
                for file_info in files_to_delete:
                    try:
                        os.remove(file_info["path"])
                        logger.info(f"Supprimé: {file_info['name']}")
                    except Exception as e:
                        logger.error(f"Erreur lors de la suppression de {file_info['path']}: {str(e)}")
                
                logger.info(f"{len(files_to_delete)} fichiers supprimés avec succès")
            else:
                logger.info("Opération annulée")
    else:
        logger.info("Aucun fichier à supprimer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nettoie les fichiers CSV obsolètes")
    parser.add_argument("-d", "--days", type=int, default=7, help="Âge maximum des fichiers en jours (défaut: 7)")
    parser.add_argument("-f", "--force", action="store_true", help="Supprimer sans confirmation")
    parser.add_argument("-k", "--keep-latest", action="store_true", help="Conserver le dernier fichier pour chaque ID vidéo")
    args = parser.parse_args()
    
    clean_cache(days=args.days, force=args.force, keep_latest=args.keep_latest) 