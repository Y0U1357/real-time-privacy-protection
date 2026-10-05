"""Export the existing MOT17 checkpoint and verify it before browser use.

Run from the repository root: python tools/export_yolox_onnx.py --image FILE
Only use trusted PyTorch checkpoints (torch.load deserializes Python objects).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model

SIZE = (544, 960)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def iou(box, other):
    inter_w = min(box[2], other[2]) - max(box[0], other[0])
    inter_h = min(box[3], other[3]) - max(box[1], other[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = ((box[2] - box[0]) * (box[3] - box[1]) +
             (other[2] - other[0]) * (other[3] - other[1]) - inter)
    return inter / union if union > 0 else 0.0


def detections(raw, ratio, conf=0.05, nms=0.7):
    """Same decode as web/js/yolox.js: the graph already decodes the head, so a
    row is [cx, cy, w, h, obj, person] in network pixels."""
    pred = raw.reshape(-1, 6)
    scores = (pred[:, 4] * pred[:, 5]).astype(np.float32)
    keep = scores >= conf
    pred, scores = pred[keep], scores[keep]
    if len(pred) == 0:
        return np.empty((0, 5), dtype=np.float32)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) / ratio
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    survivors = []
    for index in range(len(boxes)):
        if all(iou(boxes[index], boxes[kept]) <= nms for kept in survivors):
            survivors.append(index)
    return np.concatenate([boxes[survivors], scores[survivors, None]], axis=1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained/bytetrack_s_mot17.pth.tar")
    parser.add_argument("--output", type=Path, default=ROOT / "web/models/yolox_s_mot17.onnx")
    parser.add_argument("--image", type=Path, help="Local person image for coordinate parity; never uploaded")
    args = parser.parse_args()
    torch.set_num_threads(4)
    exp = get_exp(str(ROOT / "exps/example/mot/yolox_s_mix_det.py"), None)
    exp.test_size, exp.test_conf = SIZE, 0.05
    model = exp.get_model().eval().float()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model = fuse_model(model).eval()
    # Keep the stock YOLOX behaviour: the head decodes inside the graph, so the
    # ONNX output is [cx, cy, w, h, obj, person] in network pixels. The raw
    # (l, t) variant would need postprocess(), which this fork mis-decodes.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, *SIZE)
    with torch.inference_mode():
        torch.onnx.export(model, sample, str(args.output), input_names=["images"],
                          output_names=["output"], opset_version=17, dynamo=False)
    onnx.checker.check_model(str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])
    images = [("synthetic", np.random.default_rng(42).integers(0, 256, (720, 1280, 3), dtype=np.uint8))]
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f"Cannot read {args.image}")
        images.append(("person_image", image))
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "input": [1, 3, *SIZE], "output": [1, 10710, 6], "dtype": "float32",
              "mean": MEAN, "std": STD, "padding": 114, "placement": "top-left",
              "strides": [8, 16, 32], "confidence": 0.05, "nms": 0.7,
              "decoded": True, "fused": True, "opset": 17, "checks": []}
    for label, image in images:
        tensor, ratio = preproc(image, SIZE, MEAN, STD)
        tensor = tensor[None]
        with torch.inference_mode():
            reference = model(torch.from_numpy(tensor)).numpy()
        actual = session.run(None, {"images": tensor})[0]
        np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-4)
        a, b = detections(reference, ratio), detections(actual, ratio)
        # Sorting by confidence gives deterministic correspondence away from ties.
        a, b = a[np.argsort(-a[:, 4])], b[np.argsort(-b[:, 4])]
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=0.1)
        report["checks"].append({"case": label, "max_raw_error": float(np.max(np.abs(actual-reference))),
                                 "people_candidates": len(a), "max_box_error_px": float(np.max(np.abs(a[:, :4]-b[:, :4]))) if len(a) else 0})
        if label == "person_image":
            fixture = args.output.parent / "validation"
            fixture.mkdir(exist_ok=True)
            cv2.imwrite(str(fixture / "image.png"), image)
            tensor.tofile(fixture / "input.f32")
            actual.tofile(fixture / "output.f32")
            (fixture / "expected.json").write_text(json.dumps({"width": image.shape[1], "height": image.shape[0],
                "ratio": ratio, "detections": b.tolist()}), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
