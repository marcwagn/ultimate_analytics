from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

class ModelDetect():
    def __init__(self, model_path):
        """
        Initializes the ModelDetect class.

        Parameters:
        - model_path (str): The path to the model file.
        
        Returns:
        - None
        """
        self.model = YOLO(model_path)

    
    def default_predict(self, img: np.ndarray) -> np.ndarray:
        """
        Perform a default prediction on the given image.

        Parameters:
            img (np.ndarray): The input image for prediction.

        Returns:
            np.ndarray: The prediction results.
        """
        results = self.model.predict(img)
        return results[0].plot()

    def team_predict(self, img: np.ndarray) -> np.ndarray:
        """
        Perform a team prediction on the given image.

        Parameters:
            img (np.ndarray): The input image for prediction.

        Returns:
            np.ndarray: The prediction results.
        """
        results = self.model.predict(img)

        pred_img = img.copy()

        player_imgs = []
        for result in results:
            for cls, box in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()):
                if cls == 0:
                    #crop player boxes
                    cropped_img = Image.fromarray(img).crop(box).resize((128, 256))
                    cv2.rectangle(pred_img, 
                                  (int(box[0]), int(box[1])), 
                                  (int(box[2]), int(box[3])), 
                                  (255, 255, 255), 2)
                    player_imgs.append(cropped_img)
         
        return pred_img