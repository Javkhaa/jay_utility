import glob
import math
import os
import shutil
import tempfile
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import pandas as pd
from loguru import logger
import torch
from ultralytics import YOLO
import yaml


def get_obb_coordinates(
    bboxes: list[torch.Tensor | np.ndarray],
) -> list[np.ndarray]:
    """Extract oriented bounding box coordinates from model predictions.

    Args:
        bboxes (list[torch.Tensor | numpy.ndarray]): Bounding box predictions in
            ``[x1, y1, x2, y2, x3, y3, x4, y4]`` order.

    Returns:
        list[numpy.ndarray]: Each element contains four ``(x, y)`` corner points.
    """
    bbox_pos = []
    for bbox in bboxes:
        if isinstance(bbox, torch.Tensor):
            rectangle = bbox.cpu().numpy().reshape((-1, 2))
        else:
            rectangle = bbox.numpy().reshape((-1, 2))
        bbox_pos.append(rectangle)

    logger.debug(f"Found {len(bbox_pos)} bounding boxes")
    return bbox_pos


def get_bbox_coordinates(
    bboxes: list[torch.Tensor | np.ndarray], width: int, height: int
) -> list[list[float]]:
    """Extract normalized bounding box coordinates from model predictions.

    Args:
        bboxes (list[torch.Tensor | numpy.ndarray]): Bounding boxes in
            ``[xcenter, ycenter, width, height]`` format.
        width (int): Width of the source image in pixels.
        height (int): Height of the source image in pixels.

    Returns:
        list[list[float]]: ``[x_center, y_center, width, height]`` ratios.
    """
    bbox_pos = []
    for bbox in bboxes:
        xcenter, ycenter, xwidth, ywidth = bbox.tolist()
        rel_xcenter = xcenter / width
        rel_ycenter = ycenter / height
        rel_width = xwidth / width
        rel_height = ywidth / height
        bbox_pos.append([rel_xcenter, rel_ycenter, rel_width, rel_height])
    return bbox_pos


def get_result_coordinates(
    results: list,
    use_normalized_coordinates: bool = False,
    max_frames: int = -1,
) -> list[list[np.ndarray]]:
    """Extract bounding box coordinates from model predictions.

    Args:
        results (list): Sequence of YOLO tracking results.
        use_normalized_coordinates (bool, optional): If ``True`` return normalized
            coordinates. Defaults to ``False``.
        max_frames (int, optional): Process at most this many frames. ``-1``
            means no limit. Defaults to ``-1``.

    Returns:
        list[list[numpy.ndarray]]: Bounding boxes for each frame.
    """
    bbox_pos_list = []
    for idx, result in enumerate(results):
        if max_frames != -1 and idx >= max_frames:
            break
        if use_normalized_coordinates and result.obb.xyxyxyxyn is not None:
            bbox_pos = get_obb_coordinates(result.obb.xyxyxyxyn)
        elif result.obb.xyxyxyxy is not None:
            bbox_pos = get_obb_coordinates(result.obb.xyxyxyxy)
        elif result.boxes.xywh is not None:
            bbox_pos = get_bbox_coordinates(
                result.boxes.xywh, result.width, result.height
            )
        bbox_pos_list.append(bbox_pos)

    return bbox_pos_list


def get_initial_prediction(
    images: list[str],
    model_path: str,
    conf: float = 0.5,
    device: str = "cuda",
    persist: bool = False,
    predict: bool = False,
    use_normalized_coordinates: bool = True,
) -> dict[str, list[list[float]]]:
    """Run YOLO model on images and return bounding box coordinates.

    Args:
        images (list[str]): Paths to images for inference.
        model_path (str): Path to the YOLO model weights.
        conf (float, optional): Confidence threshold. Defaults to ``0.5``.
        device (str, optional): Device used for inference. Defaults to ``"cuda"``.
        persist (bool, optional): Whether to persist tracking state. Defaults to ``False``.
        predict (bool, optional): If ``True``, run prediction instead of tracking. Defaults to ``False``.
        use_normalized_coordinates (bool, optional): Return normalized coordinates. Defaults to ``True``.

    Returns:
        dict[str, list[list[float]]]: Mapping of image path to bounding box coordinates.
    """
    model = YOLO(model_path)
    # Run batched inference on a list of images
    if predict:
        results = model.predict(images, conf=conf, device=device)
    else:
        results = model.track(
            images, conf=conf, device=device, persist=persist, stream=True
        )
    realized_result = {}
    bbox_coordinates = get_result_coordinates(results, use_normalized_coordinates=use_normalized_coordinates)
    # Process results list
    for img_file, result in zip(images, bbox_coordinates):
        # result should contain a list of coordinates for each object in the image
        realized_result[img_file] = result

    return realized_result


def create_cvat_importable_annotations(
    image_dir: str,
    model_path: str,
    outdir: str,
    conf: float = 0.5,
    label_name: str = "shrimp",
    max_images: int | None = None,
) -> str:
    """Create CVAT-importable annotations using a YOLO model.

    Args:
        image_dir (str): Directory containing source images.
        model_path (str): Path to the YOLO model weights.
        outdir (str): Directory where the dataset will be created.
        conf (float, optional): Confidence threshold. Defaults to ``0.5``.
        label_name (str, optional): Name of the label class. Defaults to ``"shrimp"``.
        max_images (int | None, optional): Limit on number of images to process. Defaults to ``None``.

    Returns:
        str: Path to the generated ZIP file.
    """
    images = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.PNG"))
    )
    if max_images != -1:
        images = images[:max_images]

    logger.info(f"Annotating {len(images)} images")

    os.makedirs(outdir, exist_ok=True)

    base_train_str = "Train"
    # create data.yaml file
    data_yaml = {
        "train": "images/train",
        "names": {0: label_name},
        "path": ".",
    }

    with open(os.path.join(outdir, "data.yaml"), "w") as fp:
        yaml.dump(data_yaml, fp)

    # Move images to images/train directory
    image_outdir = os.path.join(outdir, "images/train")
    os.makedirs(image_outdir, exist_ok=True)
    for image in images:
        shutil.copy(image, os.path.join(image_outdir, os.path.basename(image)))

    realized_coordinates = get_initial_prediction(
        images=images, model_path=model_path, conf=conf, use_normalized_coordinates=True
    )
    # Write labels to labels/train directory
    labels_outdir = os.path.join(outdir, "labels/train")
    os.makedirs(labels_outdir, exist_ok=True)
    for img_file, result in realized_coordinates.items():
        label_file = os.path.join(
            labels_outdir, os.path.basename(img_file).replace(".png", ".txt")
        )
        with open(label_file, "w") as fp:
            for box_coords in result:
                coord_str = ""
                for coord in box_coords:
                    coord_str += f"{coord[0]} {coord[1]} "
                fp.write(f"0 {coord_str}\n")

    # Zip the outdir in its parent directory
    parent_dir = os.path.dirname(outdir)
    outdir_name = os.path.basename(outdir)
    zip_file_name = os.path.join(parent_dir, outdir_name)
    shutil.make_archive(
        zip_file_name,
        "zip",
        parent_dir,
        outdir_name,
    )
    zip_file_name = zip_file_name + ".zip"
    logger.info(f"Created cvat importable annotations at {zip_file_name}")
    return zip_file_name


def show_tracked_video(images: list[str], bboxes: list[list[torch.Tensor]]) -> None:
    """Display tracked bounding boxes over a sequence of images.

    Args:
        images (list[str]): Paths to image files.
        bboxes (list[list[torch.Tensor]]): Bounding boxes for each frame.
    """
    cv2.startWindowThread()
    tracking_frame = defaultdict(list)
    for idx, frame_bbox in enumerate(bboxes):
        img_file = images[idx]
        basename = os.path.basename(img_file)

        frame = cv2.imread(img_file)

        width, height, _ = frame.shape
        for idx, bbox in enumerate(frame_bbox):
            xcenter, ycenter, xwidth, ywidth = [int(elem) for elem in bbox.tolist()]
            x1 = xcenter - xwidth // 2
            y1 = ycenter - ywidth // 2
            cv2.rectangle(frame, (x1, y1), (x1 + xwidth, y1 + ywidth), (255, 0, 0), 2)
            tracking_frame[idx].append([xcenter, ycenter])
            for path_idx, curr_line_center in enumerate(tracking_frame[idx]):
                cv2.circle(frame, curr_line_center, 5, (0, 0, 255), -1)
                prev_line_center = tuple(tracking_frame[idx][path_idx - 1])
                if (
                    path_idx > 0
                    and math.sqrt(
                        (prev_line_center[0] - curr_line_center[0]) ** 2
                        + (prev_line_center[1] - curr_line_center[1]) ** 2
                    )
                    < 50
                ):
                    cv2.line(frame, prev_line_center, curr_line_center, (0, 0, 255), 5)
            if len(tracking_frame[idx]) > 10:
                tracking_frame[idx].pop(0)

        if cv2.waitKey(33) & 0xFF == ord(
            "q"
        ):  # Wait for specified ms or until 'q' is pressed
            break

        cv2.imshow("frame", frame)
        idx += 1

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def get_object_speeds(
    results: dict[str, Any],
    speed_thresholds: dict[str, float],
) -> pd.DataFrame:
    """Summarize object movement speed from tracking results.

    Args:
        results (dict[str, Any]): Mapping of frame name to YOLO result object.
        speed_thresholds (dict[str, float]): Thresholds defining ``"slow"`` and ``"fast"``.

    Returns:
        pandas.DataFrame: Descriptive statistics of object speeds with color codes.
    """
    obj_locations = defaultdict(list)
    obj_frame_locations = []
    for _, result in results.items():
        obj_frame_locations.append(result)
        if result.boxes.id is None:
            continue
        for idx, obj_id in enumerate(result.boxes.id):
            obj_locations[int(obj_id)].append(result.boxes.xywh[idx].tolist())

    obj_dist_delta = {}
    for obj_id, locations in obj_locations.items():
        loc_df = pd.DataFrame(locations)
        loc_df.columns = ["xcenter", "ycenter", "width", "height"]
        loc_df["x_delta"] = loc_df["xcenter"].diff()
        loc_df["y_delta"] = loc_df["ycenter"].diff()
        loc_df["dist_delta"] = np.sqrt(loc_df["x_delta"] ** 2 + loc_df["y_delta"] ** 2)
        obj_dist_delta[obj_id] = loc_df["dist_delta"]

    obj_dist_delta = pd.DataFrame.from_dict(obj_dist_delta)
    obj_dist_delta_summary = obj_dist_delta.describe().T
    obj_dist_delta_summary["color"] = obj_dist_delta_summary.apply(
        lambda x: get_speed_color(x["mean"], speed_thresholds), axis=1
    )
    return obj_dist_delta_summary


def get_speed_color(speed: float, speed_thresholds: dict[str, float]) -> int:
    """Return a color code representing speed category.

    Args:
        speed (float): Average movement speed.
        speed_thresholds (dict[str, float]): Thresholds with keys ``"slow"`` and ``"fast"``.

    Returns:
        int: Color index indicating movement category.
    """
    if speed <= speed_thresholds["slow"]:
        return 6  # red not moving much
    elif speed <= speed_thresholds["fast"]:
        return 2  # white normal movement
    elif speed > speed_thresholds["fast"]:
        return 0  # blue fast

    return 3  # aquamarine for unknown


class TemporaryDirectory:
    """Context manager that creates and cleans up a temporary directory."""

    def __enter__(self) -> str:
        """Create the temporary directory."""
        self.dir = tempfile.mkdtemp()
        return self.dir

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Remove the temporary directory."""
        shutil.rmtree(self.dir)


def create_video_from_images(frame_folder: str, output_video_path: str) -> None:
    """Create a video from images in a folder.

    Args:
        frame_folder (str): Directory containing frame images.
        output_video_path (str): Path to the output video file.
    """
    frames = [f for f in os.listdir(frame_folder) if f.endswith(".png")]
    frames.sort(
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )  # Sorting numerically based on frame number

    # Read the first frame to get frame size
    frame = cv2.imread(os.path.join(frame_folder, frames[0]))
    height, width, layers = frame.shape

    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )

    for frame in frames:
        img = cv2.imread(os.path.join(frame_folder, frame))
        out.write(img)  # Write each frame to the video

    out.release()
    cv2.destroyAllWindows()


def generate_speed_color_coded_video(
    results: dict[str, Any],
    output_video_path: str,
    speed_thresholds: dict[str, float] = {"slow": 5, "fast": 10},
) -> pd.DataFrame:
    """Generate a video with objects color-coded by speed.

    Args:
        results (dict[str, Any]): Mapping of frame file names to YOLO results.
        output_video_path (str): Destination for the generated video.
        speed_thresholds (dict[str, float], optional): Speed thresholds for color coding.

    Returns:
        pandas.DataFrame: Summary statistics of object speeds.
    """
    obj_dist_delta_summary = get_object_speeds(results, speed_thresholds)
    speed_color = obj_dist_delta_summary["color"].to_dict()
    # Write annotated images into temp directory
    with TemporaryDirectory() as temp_dir:
        for frame_idx, (fname, result) in enumerate(results.items()):
            basename = os.path.basename(fname)
            annot_fname = os.path.join(
                temp_dir, basename.replace(".png", f"_frame_{frame_idx:04d}.png")
            )
            result.plot(
                save=True,
                filename=annot_fname,
                line_width=2,
                custom_color_ids=speed_color,
            )

        create_video_from_images(temp_dir, output_video_path)
    return obj_dist_delta_summary


def get_rectangle_area(coords: list[float]) -> float:
    """Compute the area of a rectangle from vertex coordinates.

    Args:
        coords (list[float]): Coordinates formatted as
            ``[x1, y1, x2, y2, x3, y3, x4, y4, ...]``.

    Returns:
        float: Area of the rectangle.
    """
    x1, y1, x2, y2, x3, y3 = coords[:6]
    width = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    height = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    return width * height
