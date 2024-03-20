import supervisely as sly
import typing as t
import numpy as np

import os
import logging

from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
from functools import partial

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

    def download_image(img: sly.api.image_api.ImageInfo, api:sly.api) -> np.ndarray:
        """ Download image from Supervisely """
        logger.info(f"Start downloading image {img.name} from Supervisely")
        return Image.fromarray(api.image.download_np(img.id))
    
    def download_annotation(img: sly.api.image_api.ImageInfo, api:sly.api)-> t.Dict[str, t.Any]:
        """ Download annotation from Supervisely """
        logger.info(f"Start downloading annotation for image {img.name} from Supervisely")
        return api.annotation.download(img.id).annotation

    if sly.is_development():
        load_dotenv(os.path.expanduser("~/supervisely.env"))

    api = sly.Api.from_env()

    metadata = api.project.get_meta(params["project_id"])

    annotations_dict = {}
    images_dict = {}

    for dataset_id in params["dataset_ids"]:
        
        image_info_list = api.image.get_list(dataset_id)

        for image in image_info_list:
            image_name = Path(image.name).stem
            images_dict[image_name] = partial(download_image, image, api)
            annotations_dict[image.name] = partial(download_annotation, image, api)

    return metadata, annotations_dict, images_dict