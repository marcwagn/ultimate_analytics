import numpy as np
import typing as t
import cv2

from ultralytics import YOLO

from PIL import Image
from sklearn.cluster import KMeans

class ImageHandler():
    def __init__(self, model_path: str, img: np.ndarray) -> None:
        """
        Initializes the ImageHandler class.

        Parameters:
        - model_path (str): The path to the model file.
        - img (np.ndarray): The input image.
        """
        self.model = YOLO(model_path)
        self.img = img
        # get model results list
        self.results = self.model.predict(img)

        # get player results list
        self.player_info = []
        for result in self.results:
            for cls, box in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()):
                if cls == 0:
                    #crop player boxes
                    cropped_img = Image.fromarray(self.img).crop(box).resize((128, 256))
                    self.player_info.append({"img": cropped_img, 
                                             "box": box})


    
    def predict(self) -> np.ndarray:
        """
        Perform a default prediction on image.

        Returns:
            np.ndarray: The prediction results as image.
        """
        return self.results[0].plot()

    def predict_teams(self) -> np.ndarray:
        """
        Perform a team prediction on image.

        Returns:
            np.ndarray: The prediction results.
        """
            
        player_imgs = self.get_player_images()
        player_labels = self.predict_player_clusters(player_imgs)

        team_img = np.copy(self.img)

        for player, label in zip(self.player_info, player_labels):
            box = player["box"]
            if label == 0:
                cv2.rectangle(team_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0), 3)
            if label == 1:
                cv2.rectangle(team_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 3)

        return team_img

    def get_player_images(self) -> t.List[Image.Image]:
        """
        Get list of player images.

        Returns:
            List[Image.Image]: A list of player images.
        """
    
        return [player["img"] for player in self.player_info]
    
    def predict_player_clusters(self, player_imgs: t.List[Image.Image]) -> t.List[int]:
        """
        Predicts the team labels for a list of player images.

        Args:
            player_imgs (List[Image.Image]): A list of player images.

        Returns:
            List[int]: A list of predicted team labels for each player image.
        """
         
        player_imgs_format = np.array([np.array(img).flatten() for img in player_imgs])

        kmeans = KMeans(n_clusters=2)
        pred_labels_kmeans = kmeans.fit_predict(player_imgs_format)
    
        if len(player_imgs_format) > 14:
            return pred_labels_kmeans
        
        #get distence for each sample to each cluster center
        cluster_distances = kmeans.transform(player_imgs_format)

        samples_class_1 = (pred_labels_kmeans == 1)
        samples_class_0 = (pred_labels_kmeans == 0)

        all_idxs = np.arange(len(player_imgs_format))

        # check if class 1 has more than 7 samples
        if np.sum(samples_class_1) > 7:
            class_1_kmeans_idxs = all_idxs[samples_class_1]
            class_1_selected_indices = self._select_n_closest_samples(1, class_1_kmeans_idxs, cluster_distances,7)
            class_0_selected_indices = np.setdiff1d(all_idxs, class_1_selected_indices)
        # check if class 0 has more than 7 samples
        elif np.sum(samples_class_0) > 7:
            class_0_kmeans_idxs = all_idxs[samples_class_0]
            class_0_selected_indices = self._select_n_closest_samples(0, class_0_kmeans_idxs, cluster_distances,7)
            class_1_selected_indices = np.setdiff1d(all_idxs, class_0_selected_indices)
        else:
            class_1_selected_indices = np.where(samples_class_1)[0]
            class_0_selected_indices = np.where(samples_class_0)[0]

        pred_labels = np.empty(len(player_imgs))
        pred_labels[class_1_selected_indices] = 1
        pred_labels[class_0_selected_indices] = 0

        return pred_labels.tolist()

    def _select_n_closest_samples(self, class_id, class_indices, cluster_distances, n):
        """
        Selects the n closest samples to the cluster center of the given class.

        Parameters:
        class_id (int): The ID of the class.
        class_indices (numpy.ndarray): An array of indices corresponding to the samples of the given class.
        cluster_distances (numpy.ndarray): A 2D array of distances between each sample and the cluster centers.
        n (int): The number of closest samples to select.

        Returns:
        numpy.ndarray: An array of indices corresponding to the selected closest samples.
        """
        class_indices_sorted_by_distance = class_indices[np.argsort(cluster_distances[:, class_id][class_indices])]
        selected_indices = class_indices_sorted_by_distance[:n]

        return selected_indices