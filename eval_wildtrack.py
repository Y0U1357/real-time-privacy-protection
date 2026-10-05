import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# ============================================================
# NumPy 舊版相容
# ============================================================

if not hasattr(np, "float"):
    np.float = float

if not hasattr(np, "int"):
    np.int = int


# ============================================================
# ByteTrack Root
# ============================================================

BYTE_TRACK_ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(BYTE_TRACK_ROOT)
)


# ============================================================
# YOLOX
# ============================================================

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess, fuse_model


# ============================================================
# 路徑
# ============================================================

# WildTrack root can be overridden with the WILDTRACK_ROOT environment variable.
WILDTRACK_ROOT = Path(os.environ.get(
    "WILDTRACK_ROOT", BYTE_TRACK_ROOT / "datasets" / "Wildtrack"
))


# C1 圖片
CAMERA_NAME = "C1"

IMAGE_DIR = (
    WILDTRACK_ROOT
    / "Image_subsets"
    / CAMERA_NAME
)


# WildTrack GT JSON
ANNOTATION_DIR = (
    WILDTRACK_ROOT
    / "annotations_positions"
)


# C1 對應 viewNum = 0
VIEW_INDEX = 0


# ============================================================
# YOLOX-S MOT17
# ============================================================

EXP_FILE = (
    BYTE_TRACK_ROOT
    / "exps"
    / "example"
    / "mot"
    / "yolox_s_mix_det.py"
)


CHECKPOINT = (
    BYTE_TRACK_ROOT
    / "pretrained"
    / "bytetrack_s_mot17.pth.tar"
)


# ============================================================
# Detection 評估設定
# ============================================================

# YOLOX 最終 detection confidence
#
# 注意：
# YOLOX Exp 內部 test_conf 通常很低，
# 因為 ByteTrack 希望保留低 confidence detections。
#
# 但現在我們是在做純 Detection Precision / Recall，
# 因此在 postprocess 後再自行過濾。
DETECTION_CONF = 0.25


# TP 判定 IoU
IOU_THRESHOLD = 0.50


# Privacy box 向四周擴張 5%
PRIVACY_PADDING = 0.05


# ============================================================
# GPU
# ============================================================

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


USE_FP16 = torch.cuda.is_available()

FUSE_MODEL = True


# ============================================================
# Error Images
# ============================================================

SAVE_ERROR_IMAGES = True

MAX_FN_IMAGES = 100

MAX_FP_IMAGES = 100


OUTPUT_DIR = (
    BYTE_TRACK_ROOT
    / "evaluation_output"
    / (
        f"C1_yoloxs_mot17"
        f"_conf{DETECTION_CONF}"
        f"_padding{int(PRIVACY_PADDING * 100)}"
    )
)


FN_DIR = OUTPUT_DIR / "false_negative"

FP_DIR = OUTPUT_DIR / "false_positive"


# ============================================================
# IoU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b


    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    inter_x1 = max(
        ax1,
        bx1
    )

    inter_y1 = max(
        ay1,
        by1
    )

    inter_x2 = min(
        ax2,
        bx2
    )

    inter_y2 = min(
        ay2,
        by2
    )


    inter_width = max(
        0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1
    )


    intersection = (
        inter_width
        *
        inter_height
    )


    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )


    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union <= 0:
        return 0.0


    return (
        intersection
        /
        union
    )


# ============================================================
# Intersection / GT Area
#
# Privacy Coverage 使用
# ============================================================

def calculate_coverage(
    gt_box,
    privacy_box
):

    gx1, gy1, gx2, gy2 = gt_box

    px1, py1, px2, py2 = privacy_box


    inter_x1 = max(
        gx1,
        px1
    )

    inter_y1 = max(
        gy1,
        py1
    )

    inter_x2 = min(
        gx2,
        px2
    )

    inter_y2 = min(
        gy2,
        py2
    )


    inter_width = max(
        0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1
    )


    intersection = (
        inter_width
        *
        inter_height
    )


    gt_area = max(
        0,
        gx2 - gx1
    ) * max(
        0,
        gy2 - gy1
    )


    if gt_area <= 0:
        return 0.0


    return min(
        1.0,
        intersection
        /
        gt_area
    )


# ============================================================
# Privacy Padding
# ============================================================

def expand_box(
    box,
    frame_width,
    frame_height,
    padding
):

    x1, y1, x2, y2 = box


    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )


    pad_x = (
        width
        *
        padding
    )

    pad_y = (
        height
        *
        padding
    )


    new_x1 = max(
        0,
        x1 - pad_x
    )

    new_y1 = max(
        0,
        y1 - pad_y
    )

    new_x2 = min(
        frame_width,
        x2 + pad_x
    )

    new_y2 = min(
        frame_height,
        y2 + pad_y
    )


    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2
    )


# ============================================================
# Load WildTrack GT
# ============================================================

def load_ground_truth(
    json_path,
    view_index
):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )


    gt_boxes = []


    # WildTrack JSON:
    #
    # [
    #     {
    #         personID: ...
    #         views: [
    #             {
    #                 viewNum: 0,
    #                 xmin: ...,
    #                 ymin: ...,
    #                 xmax: ...,
    #                 ymax: ...
    #             }
    #         ]
    #     }
    # ]
    #
    for person in data:

        views = person.get(
            "views",
            []
        )


        for view in views:

            if (
                view.get(
                    "viewNum"
                )
                != view_index
            ):
                continue


            xmin = view.get(
                "xmin",
                -1
            )

            ymin = view.get(
                "ymin",
                -1
            )

            xmax = view.get(
                "xmax",
                -1
            )

            ymax = view.get(
                "ymax",
                -1
            )


            # WildTrack 看不到的人通常是 -1
            if (
                xmin < 0
                or ymin < 0
                or xmax < 0
                or ymax < 0
            ):
                continue


            if (
                xmax <= xmin
                or ymax <= ymin
            ):
                continue


            gt_boxes.append(
                (
                    float(xmin),
                    float(ymin),
                    float(xmax),
                    float(ymax)
                )
            )


    return gt_boxes


# ============================================================
# Greedy One-to-One Matching
# ============================================================

def match_predictions(
    gt_boxes,
    pred_boxes,
    iou_threshold
):

    candidates = []


    # --------------------------------------------------------
    # 找出所有 GT / Prediction pair
    # --------------------------------------------------------

    for gt_index, gt_box in enumerate(
        gt_boxes
    ):

        for pred_index, pred_box in enumerate(
            pred_boxes
        ):

            iou = calculate_iou(
                gt_box,
                pred_box
            )


            if (
                iou
                >= iou_threshold
            ):

                candidates.append(
                    (
                        iou,
                        gt_index,
                        pred_index
                    )
                )


    # --------------------------------------------------------
    # IoU 最大的優先配對
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )


    matched_gt = set()

    matched_pred = set()

    matches = []


    for (
        iou,
        gt_index,
        pred_index
    ) in candidates:

        if (
            gt_index
            in matched_gt
        ):
            continue


        if (
            pred_index
            in matched_pred
        ):
            continue


        matched_gt.add(
            gt_index
        )

        matched_pred.add(
            pred_index
        )


        matches.append(
            (
                gt_index,
                pred_index,
                iou
            )
        )


    # --------------------------------------------------------
    # FN
    # --------------------------------------------------------

    unmatched_gt = [

        i

        for i in range(
            len(gt_boxes)
        )

        if i not in matched_gt
    ]


    # --------------------------------------------------------
    # FP
    # --------------------------------------------------------

    unmatched_pred = [

        i

        for i in range(
            len(pred_boxes)
        )

        if i not in matched_pred
    ]


    return (
        matches,
        unmatched_gt,
        unmatched_pred
    )


# ============================================================
# Draw Error Image
# ============================================================

def draw_error_image(
    image,
    gt_boxes,
    pred_boxes,
    matches,
    unmatched_gt,
    unmatched_pred
):

    output = image.copy()


    matched_gt = {
        x[0]
        for x in matches
    }

    matched_pred = {
        x[1]
        for x in matches
    }


    # ========================================================
    # GT
    # ========================================================

    for i, box in enumerate(
        gt_boxes
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )


        if i in matched_gt:

            # Green = matched GT
            color = (
                0,
                255,
                0
            )

            label = "GT TP"

        else:

            # Yellow = FN
            color = (
                0,
                255,
                255
            )

            label = "GT FN"


        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # Prediction
    # ========================================================

    for i, box in enumerate(
        pred_boxes
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )


        if i in matched_pred:

            # Blue = TP prediction
            color = (
                255,
                0,
                0
            )

            label = "PRED TP"

        else:

            # Red = FP prediction
            color = (
                0,
                0,
                255
            )

            label = "PRED FP"


        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        cv2.putText(
            output,
            label,
            (
                x1,
                min(
                    output.shape[0] - 5,
                    y2 + 15
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )


    return output


# ============================================================
# YOLOX Predictor
# ============================================================

class Predictor:

    def __init__(
        self,
        model,
        exp
    ):

        self.model = model

        self.num_classes = (
            exp.num_classes
        )


        # ----------------------------------------------------
        # 這裡故意沿用 Exp 原本很低的 threshold
        #
        # 最後真正的 0.25 threshold
        # 在 outputs 出來後才自行過濾。
        # ----------------------------------------------------

        self.conf_threshold = (
            exp.test_conf
        )

        self.nms_threshold = (
            exp.nmsthre
        )

        self.test_size = (
            exp.test_size
        )


        self.rgb_means = (
            0.485,
            0.456,
            0.406
        )

        self.std = (
            0.229,
            0.224,
            0.225
        )


    def predict(
        self,
        image
    ):

        original_height = (
            image.shape[0]
        )

        original_width = (
            image.shape[1]
        )


        # ====================================================
        # Preprocessing
        # ====================================================

        img, ratio = preproc(

            image,

            self.test_size,

            self.rgb_means,

            self.std

        )


        img = (
            torch
            .from_numpy(img)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )


        if USE_FP16:

            img = img.half()


        # ====================================================
        # GPU timing
        # ====================================================

        if torch.cuda.is_available():

            torch.cuda.synchronize()


        start_time = (
            time.perf_counter()
        )


        # ====================================================
        # YOLOX
        # ====================================================

        with torch.no_grad():

            outputs = self.model(
                img
            )


            outputs = postprocess(

                outputs,

                self.num_classes,

                self.conf_threshold,

                self.nms_threshold

            )


        if torch.cuda.is_available():

            torch.cuda.synchronize()


        end_time = (
            time.perf_counter()
        )


        inference_time = (
            end_time
            -
            start_time
        )


        # ====================================================
        # No detections
        # ====================================================

        if (
            outputs is None
            or len(outputs) == 0
            or outputs[0] is None
        ):

            return (
                [],
                [],
                inference_time
            )


        output = (
            outputs[0]
            .detach()
            .cpu()
            .numpy()
        )


        # ====================================================
        # YOLOX output:
        #
        # x1
        # y1
        # x2
        # y2
        # object_conf
        # class_conf
        # class_id
        #
        # person score =
        # object_conf * class_conf
        # ====================================================

        boxes = (
            output[:, 0:4]
            /
            ratio
        )


        scores = (
            output[:, 4]
            *
            output[:, 5]
        )


        pred_boxes = []

        pred_scores = []


        for (
            box,
            score
        ) in zip(
            boxes,
            scores
        ):

            if (
                float(score)
                <
                DETECTION_CONF
            ):
                continue


            x1, y1, x2, y2 = (
                box
            )


            # 防止超出原圖
            x1 = max(
                0.0,
                min(
                    float(x1),
                    original_width
                )
            )

            y1 = max(
                0.0,
                min(
                    float(y1),
                    original_height
                )
            )

            x2 = max(
                0.0,
                min(
                    float(x2),
                    original_width
                )
            )

            y2 = max(
                0.0,
                min(
                    float(y2),
                    original_height
                )
            )


            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue


            pred_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            pred_scores.append(
                float(score)
            )


        return (
            pred_boxes,
            pred_scores,
            inference_time
        )


# ============================================================
# Load Model
# ============================================================

def load_model():

    if not EXP_FILE.exists():

        raise FileNotFoundError(
            f"找不到 Exp：{EXP_FILE}"
        )


    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"找不到模型：{CHECKPOINT}"
        )


    print(
        "Loading YOLOX-S MOT17..."
    )


    exp = get_exp(
        str(EXP_FILE),
        None
    )

    # Match the browser/Python prototype evaluation settings.
    exp.test_size = (544, 960)
    exp.test_conf = 0.05
    exp.nmsthre = 0.7


    model = (
        exp
        .get_model()
        .to(DEVICE)
    )


    model.eval()


    # ========================================================
    # Checkpoint
    # ========================================================

    try:

        checkpoint = torch.load(

            str(CHECKPOINT),

            map_location="cpu",

            weights_only=False

        )

    except TypeError:

        checkpoint = torch.load(

            str(CHECKPOINT),

            map_location="cpu"

        )


    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model"
        in checkpoint
    ):

        state_dict = (
            checkpoint["model"]
        )

    else:

        state_dict = (
            checkpoint
        )


    model.load_state_dict(
        state_dict
    )


    # ========================================================
    # Fuse
    # ========================================================

    if FUSE_MODEL:

        model = fuse_model(
            model
        )


    # ========================================================
    # FP16
    # ========================================================

    if USE_FP16:

        model = model.half()


    predictor = Predictor(
        model,
        exp
    )


    return (
        predictor,
        exp
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Path Check
    # ========================================================

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"找不到 C1：{IMAGE_DIR}"
        )


    if not ANNOTATION_DIR.exists():

        raise FileNotFoundError(
            f"找不到 GT：{ANNOTATION_DIR}"
        )


    # ========================================================
    # Error folders
    # ========================================================

    if SAVE_ERROR_IMAGES:

        FN_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        FP_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


    # ========================================================
    # Load Model
    # ========================================================

    (
        predictor,
        exp
    ) = load_model()


    # ========================================================
    # Find JSON
    # ========================================================

    annotation_files = sorted(
        ANNOTATION_DIR.glob(
            "*.json"
        )
    )


    if len(annotation_files) == 0:

        raise RuntimeError(
            "annotations_positions 裡找不到 JSON"
        )


    print()

    print(
        "=" * 70
    )

    print(
        "WildTrack C1 Evaluation"
    )

    print(
        "=" * 70
    )


    print(
        "Model             : "
        "YOLOX-S MOT17"
    )


    print(
        f"Confidence        : "
        f"{DETECTION_CONF}"
    )


    print(
        f"IoU Threshold     : "
        f"{IOU_THRESHOLD}"
    )


    print(
        f"YOLOX Test Size   : "
        f"{exp.test_size}"
    )


    print(
        f"Privacy Padding   : "
        f"{PRIVACY_PADDING * 100:.0f}%"
    )


    print(
        f"Device            : "
        f"{DEVICE}"
    )

    print()


    # ========================================================
    # Warm-up
    # ========================================================

    first_image_files = sorted(
        IMAGE_DIR.glob(
            "*"
        )
    )


    first_image_files = [

        path

        for path in first_image_files

        if path.suffix.lower()
        in {
            ".png",
            ".jpg",
            ".jpeg"
        }
    ]


    if len(first_image_files) == 0:

        raise RuntimeError(
            "C1 沒有圖片"
        )


    warmup_image = cv2.imread(
        str(
            first_image_files[0]
        )
    )


    print(
        "GPU Warm-up..."
    )


    for _ in range(5):

        predictor.predict(
            warmup_image
        )


    print(
        "Warm-up complete."
    )

    print()


    # ========================================================
    # Statistics
    # ========================================================

    processed_images = 0

    skipped_images = 0


    total_gt = 0

    total_predictions = 0


    total_tp = 0

    total_fp = 0

    total_fn = 0


    total_inference_time = 0.0


    privacy_coverages = []


    saved_fn = 0

    saved_fp = 0


    # ========================================================
    # Loop
    # ========================================================

    for index, json_path in enumerate(
        annotation_files,
        start=1
    ):

        # ----------------------------------------------------
        # WildTrack annotation:
        #
        # 00000000.json
        #
        # C1:
        #
        # 00000000.png
        # ----------------------------------------------------

        stem = (
            json_path.stem
        )


        possible_images = [

            IMAGE_DIR
            / f"{stem}.png",

            IMAGE_DIR
            / f"{stem}.jpg",

            IMAGE_DIR
            / f"{stem}.jpeg"

        ]


        image_path = None


        for candidate in possible_images:

            if candidate.exists():

                image_path = (
                    candidate
                )

                break


        if image_path is None:

            skipped_images += 1

            continue


        image = cv2.imread(
            str(
                image_path
            )
        )


        if image is None:

            skipped_images += 1

            continue


        height, width = (
            image.shape[:2]
        )


        # ====================================================
        # GT
        # ====================================================

        gt_boxes = load_ground_truth(

            json_path,

            VIEW_INDEX

        )


        # ====================================================
        # Prediction
        # ====================================================

        (
            pred_boxes,
            pred_scores,
            inference_time
        ) = predictor.predict(
            image
        )


        # ====================================================
        # Matching
        # ====================================================

        (
            matches,
            unmatched_gt,
            unmatched_pred
        ) = match_predictions(

            gt_boxes,

            pred_boxes,

            IOU_THRESHOLD

        )


        tp = len(
            matches
        )

        fn = len(
            unmatched_gt
        )

        fp = len(
            unmatched_pred
        )


        # ====================================================
        # Statistics
        # ====================================================

        processed_images += 1


        total_gt += len(
            gt_boxes
        )


        total_predictions += len(
            pred_boxes
        )


        total_tp += tp

        total_fn += fn

        total_fp += fp


        total_inference_time += (
            inference_time
        )


        # ====================================================
        # Privacy Coverage
        #
        # 對每一個 GT：
        #
        # 找所有 Prediction Privacy Boxes 中
        # 覆蓋 GT 最大的那一個。
        # ====================================================

        privacy_boxes = [

            expand_box(

                pred_box,

                width,

                height,

                PRIVACY_PADDING

            )

            for pred_box in pred_boxes

        ]


        for gt_box in gt_boxes:

            best_coverage = 0.0


            for privacy_box in (
                privacy_boxes
            ):

                coverage = (
                    calculate_coverage(

                        gt_box,

                        privacy_box

                    )
                )


                if (
                    coverage
                    >
                    best_coverage
                ):

                    best_coverage = (
                        coverage
                    )


            privacy_coverages.append(
                best_coverage
            )


        # ====================================================
        # Save FN
        # ====================================================

        if (
            SAVE_ERROR_IMAGES
            and fn > 0
            and saved_fn
            < MAX_FN_IMAGES
        ):

            error_image = (
                draw_error_image(

                    image,

                    gt_boxes,

                    pred_boxes,

                    matches,

                    unmatched_gt,

                    unmatched_pred

                )
            )


            output_path = (
                FN_DIR
                /
                (
                    f"{stem}"
                    f"_FN{fn}"
                    f"_FP{fp}.jpg"
                )
            )


            cv2.imwrite(
                str(
                    output_path
                ),
                error_image
            )


            saved_fn += 1


        # ====================================================
        # Save FP
        # ====================================================

        if (
            SAVE_ERROR_IMAGES
            and fp > 0
            and saved_fp
            < MAX_FP_IMAGES
        ):

            error_image = (
                draw_error_image(

                    image,

                    gt_boxes,

                    pred_boxes,

                    matches,

                    unmatched_gt,

                    unmatched_pred

                )
            )


            output_path = (
                FP_DIR
                /
                (
                    f"{stem}"
                    f"_FP{fp}"
                    f"_FN{fn}.jpg"
                )
            )


            cv2.imwrite(
                str(
                    output_path
                ),
                error_image
            )


            saved_fp += 1


        # ====================================================
        # Progress
        # ====================================================

        if (
            processed_images
            % 25
            == 0
        ):

            print(
                f"{processed_images}/"
                f"{len(annotation_files)}"
                f" | GT={total_gt}"
                f" | Pred={total_predictions}"
                f" | TP={total_tp}"
                f" | FP={total_fp}"
                f" | FN={total_fn}"
            )


    # ========================================================
    # Metrics
    # ========================================================

    precision = (

        total_tp
        /
        (
            total_tp
            +
            total_fp
        )

        if (
            total_tp
            +
            total_fp
        ) > 0

        else 0.0
    )


    recall = (

        total_tp
        /
        (
            total_tp
            +
            total_fn
        )

        if (
            total_tp
            +
            total_fn
        ) > 0

        else 0.0
    )


    f1 = (

        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )

        if (
            precision
            +
            recall
        ) > 0

        else 0.0
    )


    miss_rate = (

        total_fn
        /
        total_gt

        if total_gt > 0

        else 0.0
    )


    average_inference = (

        total_inference_time
        /
        processed_images

        if processed_images > 0

        else 0.0
    )


    approximate_fps = (

        1.0
        /
        average_inference

        if average_inference > 0

        else 0.0
    )


    # ========================================================
    # Privacy Metrics
    # ========================================================

    privacy_array = np.asarray(
        privacy_coverages,
        dtype=np.float64
    )


    if privacy_array.size > 0:

        average_coverage = float(
            privacy_array.mean()
        )


        coverage_50 = int(
            np.sum(
                privacy_array
                >= 0.50
            )
        )


        coverage_80 = int(
            np.sum(
                privacy_array
                >= 0.80
            )
        )


        coverage_90 = int(
            np.sum(
                privacy_array
                >= 0.90
            )
        )


        coverage_zero = int(
            np.sum(
                privacy_array
                <= 0.0
            )
        )

    else:

        average_coverage = 0.0

        coverage_50 = 0

        coverage_80 = 0

        coverage_90 = 0

        coverage_zero = 0


    # ========================================================
    # Final Result
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "WildTrack C1 Evaluation"
    )

    print(
        "=" * 70
    )


    print()


    print(
        "Model             : "
        "YOLOX-S MOT17"
    )


    print(
        f"Confidence        : "
        f"{DETECTION_CONF}"
    )


    print(
        f"IoU Threshold     : "
        f"{IOU_THRESHOLD}"
    )


    print(
        f"YOLOX Test Size   : "
        f"{exp.test_size}"
    )


    print(
        f"Privacy Padding   : "
        f"{PRIVACY_PADDING * 100:.0f}%"
    )


    print()


    print(
        f"Processed Images  : "
        f"{processed_images}"
    )


    print(
        f"Skipped Images    : "
        f"{skipped_images}"
    )


    print()


    print(
        f"Ground Truth      : "
        f"{total_gt}"
    )


    print(
        f"Predictions       : "
        f"{total_predictions}"
    )


    print()


    print(
        f"TP                : "
        f"{total_tp}"
    )


    print(
        f"FP                : "
        f"{total_fp}"
    )


    print(
        f"FN                : "
        f"{total_fn}"
    )


    print()


    print(
        f"Precision         : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )


    print(
        f"Recall            : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )


    print(
        f"F1 Score          : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )


    print(
        f"Miss Rate         : "
        f"{miss_rate:.4f} "
        f"({miss_rate * 100:.2f}%)"
    )


    print()


    print(
        f"Avg Inference     : "
        f"{average_inference * 1000:.2f} ms"
    )


    print(
        f"Approx FPS        : "
        f"{approximate_fps:.2f}"
    )


    print()


    print(
        "-" * 70
    )

    print(
        "Privacy Coverage"
    )

    print(
        "-" * 70
    )


    print(
        f"Average Coverage  : "
        f"{average_coverage:.4f} "
        f"({average_coverage * 100:.2f}%)"
    )


    if total_gt > 0:

        print(
            f"Coverage >= 50% : "
            f"{coverage_50}/"
            f"{total_gt} "
            f"({coverage_50 / total_gt * 100:.2f}%)"
        )


        print(
            f"Coverage >= 80% : "
            f"{coverage_80}/"
            f"{total_gt} "
            f"({coverage_80 / total_gt * 100:.2f}%)"
        )


        print(
            f"Coverage >= 90% : "
            f"{coverage_90}/"
            f"{total_gt} "
            f"({coverage_90 / total_gt * 100:.2f}%)"
        )


        print(
            f"Coverage = 0      : "
            f"{coverage_zero}/"
            f"{total_gt} "
            f"({coverage_zero / total_gt * 100:.2f}%)"
        )


    print()


    print(
        f"Error Images      : "
        f"{OUTPUT_DIR}"
    )


    print(
        f"Saved FN Images   : "
        f"{saved_fn}"
    )


    print(
        f"Saved FP Images   : "
        f"{saved_fp}"
    )


    print()

    print(
        "=" * 70
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()