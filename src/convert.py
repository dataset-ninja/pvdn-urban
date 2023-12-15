# https://www.kaggle.com/datasets/lukasewecker/pvdn-urban

import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "PVDN urban"
    train_path = "/home/grokhi/rawdata/pvdn-urban/train/train"
    val_path = "/home/grokhi/rawdata/pvdn-urban/val/val"
    test_path = "/home/grokhi/rawdata/pvdn-urban/test/test"

    images_folder = "images"
    masks_folder = "masks"
    json_file = "annotations.json"
    masks_ext = "_mask.png"

    batch_size = 30

    ds_name_to_data = {"train": train_path, "val": val_path, "test": test_path}

    def create_ann(image_path):
        labels = []
        tags = []

        ann_data = im_name_to_tags[get_file_name_with_ext(image_path)]
        if ann_data["oncoming_vehicle_visible"] is True:
            visible = sly.Tag(visible_meta)
            tags.append(visible)
        if ann_data["contains_annotations"] is True:
            contains = sly.Tag(contains_meta)
            tags.append(contains)

        mask_path = os.path.join(masks_path, get_file_name(image_path) + masks_ext)
        mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
        img_height = mask_np.shape[0]
        img_wight = mask_np.shape[1]
        unique_pixels = np.unique(mask_np)[1:]
        for curr_pixel in unique_pixels:
            mask = mask_np == curr_pixel
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            for i in range(1, ret):
                obj_mask = curr_mask == i
                curr_bitmap = sly.Bitmap(obj_mask)
                if curr_bitmap.area > 50:
                    curr_label = sly.Label(curr_bitmap, reflection)
                    labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    reflection = sly.ObjClass("reflection", sly.Bitmap)

    visible_meta = sly.TagMeta("oncoming vehicle visible", sly.TagValueType.NONE)
    contains_meta = sly.TagMeta("contains annotations", sly.TagValueType.NONE)

    meta = sly.ProjectMeta(obj_classes=[reflection], tag_metas=[visible_meta, contains_meta])

    api.project.update_meta(project.id, meta.to_json())

    for ds_name, data_path in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_path = os.path.join(data_path, images_folder)
        images_names = os.listdir(images_path)
        masks_path = os.path.join(data_path, masks_folder)
        tags_path = os.path.join(data_path, json_file)
        im_name_to_tags = {}
        tags_data = load_json_file(tags_path)["images"]
        for curr_tag_data in tags_data:
            im_name_to_tags[curr_tag_data["file_path"].split("/")[-1]] = curr_tag_data["metadata"]

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_path, image_path) for image_path in img_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))
    return project
