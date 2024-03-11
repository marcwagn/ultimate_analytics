import supervisely as sly
import typing as t
import numpy as np

import os
import logging

from dotenv import load_dotenv
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)


def helper_download_image_dataset_from_supervisely(params: t.Dict) -> t.Tuple[t.Dict, t.Dict, t.Dict]:
    """ Download image dataset from Supervisely
    
    Args:
        params: A dictionary containing the following keys:
            - project_id: The ID of the Supervisely project
            - dataset_id: The ID of the Supervisely dataset
    Returns:
        A tuple containing:
            - metadata: The metadata of the Supervisely project
            - annotations_dict: A dictionary containing the annotations for each image
            - images_dict: A dictionary containing the images
    """
    
    if sly.is_development():
        load_dotenv(os.path.expanduser("~/supervisely.env"))

    api = sly.Api.from_env()

    metadata = api.project.get_meta(params["project_id"])

    image_info_list = api.image.get_list(params["dataset_id"])

    annotations_dict = {}
    images_dict = {}

    for image in image_info_list:

        image_name = Path(image.name).stem

        images_dict[image_name] = download_image_lazy(image, api)
        annotations_dict[image.name] = download_annotation_lazy(image, api)

    return metadata, annotations_dict, images_dict


def download_image_lazy(image: sly.api.image_api.ImageInfo, api: sly.api):
    """ Download image from Supervisely lazily
    
        Args: 
            image: The image to download
            api: The Supervisely API
        Returns:
            A function that downloads the image
    """
    def download_image() -> np.ndarray:
        logger.info(f"Start downloading image {image.name} from Supervisely")
        return Image.fromarray(api.image.download_np(image.id))
    
    return download_image


def download_annotation_lazy(image: sly.api.image_api.ImageInfo, api: sly.api):
    """ Download annotation from Supervisely lazily
    
        Args: 
            image: image for which the annotation should be downloaded
            api: The Supervisely API
        Returns:
            A function that downloads the annotation
    """
    def download_annotation()-> t.Dict[str, t.Any]:
        logger.info(f"Start downloading annotation for image {image.name} from Supervisely")
        return api.annotation.download(image.id).annotation
    
    return download_annotation
