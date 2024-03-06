import supervisely as sly
import typing as t

import os

from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
from tqdm import tqdm

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

    for image in tqdm(image_info_list):

        image_name = Path(image.name).stem

        img_data = Image.fromarray(api.image.download_np(image.id))
        images_dict[image_name] = img_data

        image_ann_json = api.annotation.download(image.id)
        annotations_dict[image_name] = image_ann_json

    return metadata, annotations_dict, images_dict
