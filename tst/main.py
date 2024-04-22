import torch

# Vérifie si CUDA est disponible
if torch.cuda.is_available():
    print("CUDA est disponible. Les modèles utiliseront le GPU.")
    device = torch.device("cuda")  # Définit le device à utiliser comme GPU
else:
    print("CUDA n'est pas disponible. Les modèles utiliseront le CPU.")
    device = torch.device("cpu")  # Définit le device à utiliser comme CPU

# Pour utiliser le device, vous pouvez envoyer vos modèles et tenseurs à ce device
# Exemple:
# model = MonAutoencodeur().to(device)
# inputs = mes_donnees.to(device)
