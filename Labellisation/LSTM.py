import numpy as np
import json
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from moviepy.editor import VideoFileClip

# Fonction pour charger les données
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        datas_fil = np.array(data['datas_fil'])
    return datas_fil

# Chargement des données
data_fil = load_data('datasForTraining.json')
data_fil = data_fil.reshape(data_fil.shape[0], data_fil.shape[1], -1)

# Division des données
X_train, X_test = train_test_split(data_fil, test_size=0.2, random_state=42)

# Définition de l'architecture du modèle
input_shape = X_train.shape[1:]
inputs = Input(shape=input_shape)
x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
latent_space = LSTM(32)(x)  # Espace latent
x = Dense(input_shape[0] * input_shape[1])(latent_space)
outputs = Reshape(input_shape)(x)
model = Model(inputs, outputs)

# Extraire le modèle de l'espace latent
latent_model = Model(inputs, latent_space)

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Évaluation du modèle
loss = model.evaluate(X_test, X_test)
print(f'Test Loss: {loss}')

# Extraire les représentations latentes
latent_train = latent_model.predict(X_train)
latent_test = latent_model.predict(X_test)

# Combinez les vecteurs latents de l'entraînement et du test
latent_vectors = np.vstack([latent_train, latent_test])

# Calculer et afficher les distances dans l'espace latent
distances = np.array([[euclidean(latent_train[i], latent_train[j]) for j in range(len(latent_train))] for i in range(len(latent_train))])

# Chemin correct du dossier vidéos
video_directory = r"C:\Users\Clément Aublet\OneDrive - ESIGELEC\Bureau\projetSeminaire\Recherche-de-similarit-et-optimisation-de-l-Arrach-\videos"

# Fonction pour trouver les vidéos les plus similaires
def find_similar_videos(latent_vectors, test_index, top_k=3):
    distances = np.array([euclidean(latent_vectors[test_index], vec) for vec in latent_vectors])
    indices = np.argsort(distances)[1:top_k+1]  # obtenir les k plus proches, sauter le premier car c'est la vidéo elle-même
    return indices, distances[indices]

# Choix une vidéo de test au hasard
test_video_index = np.random.randint(len(X_test))
similar_indices, similar_distances = find_similar_videos(latent_vectors, test_video_index + len(X_train))  # décalage par taille de train

# Affichage des vidéos similaires
print(f"Vidéo de référence (Test index {test_video_index}):")
print("Vidéos similaires trouvées:")
video_paths = sorted([os.path.join(video_directory, f) for f in os.listdir(video_directory)])
for idx, dist in zip(similar_indices, similar_distances):
    print(f"Vidéo {idx} à une distance de {dist:.3f}")
    video_clip = VideoFileClip(video_paths[idx])
    video_clip.preview()
