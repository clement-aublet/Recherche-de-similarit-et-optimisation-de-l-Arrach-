# Création du fichier json pour stocker les données
# Liste pour stocker les chemins complets des fichiers vidéo

#import ImportationVideo
import os
import re
import json

dossier_videos = 'C:/Users/Clément Aublet/OneDrive - ESIGELEC/Bureau/projetSeminaire/Recherche-de-similarit-et-optimisation-de-l-Arrach-/videos/'
fichiers_videos = [os.path.join(dossier_videos, fichier) for fichier in os.listdir(dossier_videos) if
                   fichier.endswith('.mp4')]


def extraire_metadonnees_enrichies(nom_fichier):
    pattern = re.compile(r'(?P<nom>.+) \((?P<poids_cat>\d+(?:\.\d+)?kg) (?P<pays>[^\)]+)\) (?P<description>.+)')
    match = pattern.search(nom_fichier)

    if match:
        metadata = match.groupdict()

        # Extraction supplémentaire pour le poids soulevé et le type de mouvement
        detail_match = re.search(r'(\d+)kg (Snatch|Clean & Jerk|Squat|CleanandJerk)', nom_fichier, re.IGNORECASE)
        if detail_match:
            metadata["poids_souleve"] = detail_match.group(1) + "kg"
            metadata["type_mouvement"] = detail_match.group(2)
        else:
            metadata["poids_souleve"] = "Inconnu"
            metadata["type_mouvement"] = "Inconnu"


        if "woman" in nom_fichier.lower() or "female" in nom_fichier.lower() or any(
                word in nom_fichier for word in ["Madame", "Miss", "Mrs"]):
            metadata["sexe"] = "Femme"
        elif "man" in nom_fichier.lower() or "male" in nom_fichier.lower() or any(
                word in nom_fichier for word in ["Monsieur", "Mr"]):
            metadata["sexe"] = "Homme"
        else:
            metadata["sexe"] = "Inconnu"

        return metadata
    else:
        return {}


metadonnees_enrichies = []

for fichier in fichiers_videos:
    nom_fichier = os.path.basename(fichier)
    metadata_enrichie = extraire_metadonnees_enrichies(nom_fichier)
    metadonnees_enrichies.append(metadata_enrichie)

chemin_json_enrichi = 'metadonnees_videos_enrichies.json'

with open(chemin_json_enrichi, 'w', encoding='utf-8') as fichier_json:
    json.dump(metadonnees_enrichies, fichier_json, ensure_ascii=False, indent=4)

print(f"Les métadonnées enrichies ont été sauvegardées dans '{chemin_json_enrichi}'.")
