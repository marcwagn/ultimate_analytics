import numpy as np
import pandas as pd

import cv2

from sklearn.cluster import KMeans


class TeamDetector:
    def __init__(self, img: np.ndarray, detector_results: list) -> None:
        """
        Initialize the TeamDetector object.
        Args:
            img (np.ndarray): The input image.
            detector_results (list['ultralytics.engine.results.Results']): 
        Returns:
            None
        """
        self.img = img
        self.results = detector_results
        self.player_info = []
        for result in self.results:
            for id, cls, box in zip(result.boxes.id, result.boxes.cls.tolist(), result.boxes.xyxy.tolist()):
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_img = cv2.resize(self.img[y1:y2, x1:x2], (128, 256))[30:100, 30:98]
                    self.player_info.append({"img": cropped_img, "box": box, "id": int(id)})

    def get_player_images(self) -> list[np.ndarray]:
        """
        Get list of player images.

        Returns:
            List[Image.Image]: A list of player images.
        """
        return [player["img"] for player in self.player_info]
    
    def train_kmeans(self, player_imgs: list[np.ndarray], 
                     n_colors: int = 3) -> tuple[KMeans, np.ndarray]:
        """ Train a KMeans model to identify the most representative colors (n_colors)
            in a list of images.
            Args:
                player_imgs: list of player images
                n_colors: number of colors to identify in the color palette
            Returns:
                KMeans model
                identified_palette: list of RGB colors identified as the most representative

        """

        flatten_imgs = [img.reshape(-1,3) for img in player_imgs]
        pixels_imgs = np.concatenate(flatten_imgs, axis=0)

        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels_imgs)
        identified_palette = np.array(kmeans.cluster_centers_).astype(int)

        return kmeans, identified_palette
    
    def label_image(self, kmeans: KMeans, player_img: np.ndarray) -> int:
        """Label the image with the color palette applied.

            Args:
                kmeans (KMeans): The KMeans model used for color clustering.
                player_img (np.ndarray): The image to be labeled.

            Returns:
                int: The label indicating the dominant color in the image. 
                     1 represents a bright color, 0 represents a dark color."""   
             
        player_img = player_img.reshape(-1,3)
        labels = kmeans.predict(player_img)
        identified_palette = np.array(kmeans.cluster_centers_).astype(int)

        white = [255, 255, 255]
        black = [0, 0, 0]

        def euclidean_distance(c1, c2):
            return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

        closest_to_white = min(identified_palette, key=lambda color: euclidean_distance(color, white))
        closest_to_black = min(identified_palette, key=lambda color: euclidean_distance(color, black))

        recolored_img = np.copy(player_img)
        for index in range(len(recolored_img)):
            recolored_img[index] = identified_palette[labels[index]]

        count_bright = np.count_nonzero(np.all(recolored_img == closest_to_white, axis=-1))
        count_dark = np.count_nonzero(np.all(recolored_img == closest_to_black, axis=-1))

        if count_bright > count_dark:
            return 1
        else:
            return 0
        
    def predict_player_clusters(self, kmeans: KMeans, player_imgs: list[np.ndarray]) -> pd.DataFrame:
        """
        Predict the team of each player based on the color clustering.
        Args:
            kmeans: KMeans model
            player_imgs: list of player images
        Returns:
            pd.DataFrame: A dataframe containing the player id and the predicted team.
        """
        ids = [player['id'] for player in self.player_info if 'id' in player]
        pred_labels = [self.label_image(kmeans, player_img) for player_img in player_imgs]

        return pd.DataFrame({ 'id': ids, 'pred_team': pred_labels})
