import glob
import os
import logging
import random
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mv_annot_img_and_label(
    prefix: str,
    img_file: str,
    label_file: str,
    dest_img_dir: str,
    dest_label_dir: str,
) -> None:
    """Move image and label files to target directories.

    Args:
        prefix (str): Prefix to apply to the destination file names.
        img_file (str): Path to the source image file.
        label_file (str): Path to the source label file.
        dest_img_dir (str): Directory to copy the image into.
        dest_label_dir (str): Directory to copy the label into.
    """
    new_label_file = os.path.basename(label_file).replace("frame", prefix)
    new_img_file = os.path.basename(img_file).replace("frame", prefix)
    shutil.copyfile(img_file, os.path.join(dest_img_dir, new_img_file))
    shutil.copyfile(label_file, os.path.join(dest_label_dir, new_label_file))


def cleanup_no_label_img(basedir: str) -> None:
    """Remove frames that contain no annotation.

    Args:
        basedir (str): Directory containing frame image and label files.
    """
    for label_file in sorted(glob.glob(os.path.join(basedir, "frame_*.txt"))):
        img_file = label_file.replace(".txt", ".PNG")
        try:
            pd.read_csv(label_file)
        except pd.errors.EmptyDataError:
            logger.warning(f"No annotation {label_file} removing image and label")
            os.remove(label_file)
            os.remove(img_file)


@dataclass
class CVATAnnot:
    """CVAT annotation"""

    video_name: str
    annot_dir: str
    downsample_percent: float = 1.0  # 1 for all image
    max_image_count: int = -1  # -1 for all image

    def __init__(
        self,
        video_name: str,
        annot_dir: str,
        downsample_percent: float = 1,
        max_image_count: int | None = None,
    ) -> None:
        """Initialize CVAT annotation handler.

        Args:
            video_name (str): Name of the source video.
            annot_dir (str): Directory containing annotation files.
            downsample_percent (float, optional): Portion of images to keep. Defaults to 1.
            max_image_count (int | None, optional): Maximum number of images to output. Defaults to None.
        """
        self.video_name = video_name
        self.annot_dir = annot_dir
        total_img_cnt = len(os.listdir(annot_dir)) // 2
        if max_image_count and max_image_count < total_img_cnt:
            self.downsample_percent = np.round(max_image_count / total_img_cnt, 2)
            self.max_image_count = max_image_count
        else:
            self.downsample_percent = downsample_percent

    def output_fltrd_data(self, outdir: str) -> None:
        """Output filtered annotation images and labels.

        Args:
            outdir (str): Root directory where train/ folders will be created.
        """
        dest_img_dir = os.path.join(outdir, "train/images/shrimp/")
        dest_label_dir = os.path.join(outdir, "train/labels/shrimp/")
        for dest_dir in [dest_img_dir, dest_label_dir]:
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

        for label_file in sorted(glob.glob(os.path.join(self.annot_dir, "*.txt"))):
            img_file = label_file.replace(".txt", ".PNG")
            if random.random() <= self.downsample_percent:
                mv_annot_img_and_label(
                    self.video_name, img_file, label_file, dest_img_dir, dest_label_dir
                )
