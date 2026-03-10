import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def extract_dominant_color(image: np.ndarray, k: int = 3) -> tuple:
    """
    Extrae el color dominante del torso de un jugador usando KMeans.
    
    Args:
        image: Imagen (crop) del jugador en formato BGR (OpenCV).
        k: Número de clusters para encontrar.
        
    Returns:
        Tupla con el color RGB dominante.
    """
    # 1. Recortar el centro de la imagen (aprox. la camiseta/torso)
    h, w = image.shape[:2]
    # Tomamos del 20% superior al 60% inferior (quitando cabeza y piernas)
    # y el 25% central del ancho
    torso = image[int(h*0.2):int(h*0.6), int(w*0.25):int(w*0.75)]
    
    # 2. Si el recorte es demasiado pequeño o nulo, devolver blanco por defecto
    if torso.size == 0 or torso.shape[0] < 2 or torso.shape[1] < 2:
        return (255, 255, 255)
        
    # 3. Convertir a RGB y aplantar la matriz para sklearn
    torso_rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
    pixels = torso_rgb.reshape(-1, 3)
    
    # 4. Aplicar KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels)
    
    # 5. Encontrar el cluster más frecuente (el color que más píxeles tiene)
    counts = Counter(labels)
    # Ignorar posibles picos de verde (césped de fondo) o carne/piel.
    # Una mejora real aquí es usar HSV y descartar el umbral verde,
    # pero para el primer prototipo cogemos simplemente el más repetido.
    dominant_cluster_idx = counts.most_common(1)[0][0]
    dominant_color = kmeans.cluster_centers_[dominant_cluster_idx]
    
    return tuple(map(int, dominant_color))


def assign_team(player_colors: list, k_teams: int = 2) -> list:
    """
    Dada una lista de colores rgb de N jugadores, los divide en Equipo 0 y Equipo 1.
    Nota: Porteros o árbitros (colores atípicos) pueden confundir este paso si no
    se excluyen previamente.
    """
    if not player_colors:
        return []
        
    data = np.array(player_colors)
    kmeans = KMeans(n_clusters=k_teams, random_state=42, n_init=10)
    team_labels = kmeans.fit_predict(data)
    
    return team_labels.tolist()

if __name__ == "__main__":
    print("[TEST] KMeans Team ID cargado. Listo para integrarse con detector.py")
