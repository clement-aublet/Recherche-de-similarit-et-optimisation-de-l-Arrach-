from pytube import Playlist
import cv2
import os

def telechargement_video(url, path='Labellisation\TestVideosInput'):   # création du dossier vidéos où nous stockerons ces dernières
    from pytube import YouTube
    yt = YouTube(url)
    # Choix du stream avec la résolution souhaitée ou le premier stream vidéo disponible
    ys = yt.streams.filter(progressive=True, file_extension='mp4').first()
    # Téléchargez la vidéo dans le dossier spécifié
    if not os.path.exists(path):
        os.makedirs(path)
    out_file = ys.download(output_path=path)
    print(f"La vidéo a été téléchargée et enregistrée sous {out_file}")
    return out_file

def lecture_video(video_path):
    # Utilisation de OpenCV pour lire la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la lecture de la vidéo ou fin de la vidéo")
            break

        # Affichez le frame actuel
        cv2.imshow('Frame', frame)

        # Quitter la boucle/changer de vidéos en appuiyant sur 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def telechargement_et_lecture_playlist(playlist_url, path='Labellisation\TestVideosInput'):
    playlist = Playlist(playlist_url)
    print(f"Téléchargement et lecture des vidéos de la playlist: {playlist.title}")
    for video_url in playlist.video_urls:
        print(f"Téléchargement de : {video_url}")
        try:
            video_path = telechargement_video(video_url, path)
            print(f"Lecture de : {video_path}")
            lecture_video(video_path)
        except Exception as e:
            print(f"Erreur lors du téléchargement ou de la lecture de la vidéo {video_url}: {e}") # exception pour voir si cela ne fonctionne pas

#L'URL de la playlist YouTube test
<<<<<<< Updated upstream
playlist_url = 'https://www.youtube.com/playlist?list=PL_RB_1AlbHf1V_BPtyn-TEsyTfqKv-iFM'
telechargement_et_lecture_playlist(playlist_url)
=======
playlist_url = 'https://youtube.com/playlist?list=PL_RB_1AlbHf1MVagKPksVOxyMubkX5R52&si=pNC_30PnvxqcHVFg'
playlist_url2 = 'https://youtube.com/playlist?list=PL_RB_1AlbHf1V_BPtyn-TEsyTfqKv-iFM&si=qHnSJGa1yKl1_niM'
telechargement_et_lecture_playlist(playlist_url2)
>>>>>>> Stashed changes
