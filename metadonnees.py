import os
import re
import json

# Chemin du dossier contenant les vidéos
dossier_videos = 'C:/Users/Clément Aublet/OneDrive - ESIGELEC/Bureau/projetSeminaire/Recherche-de-similarit-et-optimisation-de-l-Arrach-/videos/'
fichiers_videos = [os.path.join(dossier_videos, fichier) for fichier in os.listdir(dossier_videos) if
                   fichier.endswith('.mp4')]


def extraire_metadonnees_enrichies(nom_fichier):
    # Expression régulière pour extraire les informations de base et les descriptions
    pattern = re.compile(r'(?P<nom>.+?) (?P<poids_cat>\d+kg) (?P<pays>[🇦-🇿]+) (?P<description>.+)\.mp4')
    match = pattern.search(nom_fichier)
    if match:
        metadata = match.groupdict()
        # Extraction du poids soulevé et du type de mouvement
        detail_match = re.search(r'(\d+)kg (Snatch|Clean & Jerk|Power Clean & Jerk|Squat|C&J)', nom_fichier, re.IGNORECASE)
        if detail_match:
            metadata["poids_souleve"] = f"{detail_match.group(1)}kg"
            metadata["type_mouvement"] = detail_match.group(2)
        else:
            metadata["poids_souleve"] = "Inconnu"
            metadata["type_mouvement"] = "Inconnu"

        # Détermination du sexe basée sur le contexte du nom de fichier
        if any(word in nom_fichier.lower() for word in ["woman", "female", "madame", "miss", "mrs"]):
            metadata["sexe"] = "Femme"
        elif any(word in nom_fichier.lower() for word in ["man", "male", "monsieur", "mr"]):
            metadata["sexe"] = "Homme"
        else:
            metadata["sexe"] = "Inconnu"

        return metadata
    return {}


metadonnees_enrichies = [extraire_metadonnees_enrichies(os.path.basename(fichier)) for fichier in fichiers_videos]

# Chemin pour sauvegarder le fichier JSON
chemin_json_enrichi = 'metadonnees_videos_enrichies.json'
with open(chemin_json_enrichi, 'w', encoding='utf-8') as fichier_json:
    json.dump(metadonnees_enrichies, fichier_json, ensure_ascii=False, indent=4)

print(f"Les métadonnées enrichies ont été sauvegardées dans '{chemin_json_enrichi}'.")
