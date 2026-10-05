import cv2
import numpy as np
import scipy
import lap

from scipy.spatial.distance import cdist

from yolox.tracker import kalman_filter

import time


# ============================================================
# NumPy bbox IoU
# 取代：
#
# from cython_bbox import bbox_overlaps as bbox_ious
#
# 避免 Windows 需要 Microsoft C++ Build Tools
# ============================================================

def bbox_ious(boxes, query_boxes):
    """
    計算兩組 bounding boxes 之間的 IoU。

    boxes:
        shape = (N, 4)
        [x1, y1, x2, y2]

    query_boxes:
        shape = (K, 4)
        [x1, y1, x2, y2]

    return:
        overlaps
        shape = (N, K)
    """

    boxes = np.asarray(
        boxes,
        dtype=np.float64
    )

    query_boxes = np.asarray(
        query_boxes,
        dtype=np.float64
    )

    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros(
        (N, K),
        dtype=np.float64
    )

    # 沒有 Bounding Box
    if N == 0 or K == 0:
        return overlaps

    # --------------------------------------------------------
    # 預先計算 boxes 的面積
    # --------------------------------------------------------

    box_areas = (
        (boxes[:, 2] - boxes[:, 0] + 1)
        *
        (boxes[:, 3] - boxes[:, 1] + 1)
    )

    # --------------------------------------------------------
    # 每個 query box 與所有 boxes 計算 IoU
    # --------------------------------------------------------

    for k in range(K):

        query_box = query_boxes[k]

        query_area = (
            (query_box[2] - query_box[0] + 1)
            *
            (query_box[3] - query_box[1] + 1)
        )

        # Intersection width
        iw = (
            np.minimum(
                boxes[:, 2],
                query_box[2]
            )
            -
            np.maximum(
                boxes[:, 0],
                query_box[0]
            )
            + 1
        )

        # Intersection height
        ih = (
            np.minimum(
                boxes[:, 3],
                query_box[3]
            )
            -
            np.maximum(
                boxes[:, 1],
                query_box[1]
            )
            + 1
        )

        # 真正有交集的 box
        valid = (
            (iw > 0)
            &
            (ih > 0)
        )

        if not np.any(valid):
            continue

        intersection = (
            iw[valid]
            *
            ih[valid]
        )

        union = (
            box_areas[valid]
            +
            query_area
            -
            intersection
        )

        overlaps[
            valid,
            k
        ] = (
            intersection
            /
            union
        )

    return overlaps


# ============================================================
# Merge Matches
# ============================================================

def merge_matches(m1, m2, shape):

    O, P, Q = shape

    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix(
        (
            np.ones(len(m1)),
            (
                m1[:, 0],
                m1[:, 1]
            )
        ),
        shape=(O, P)
    )

    M2 = scipy.sparse.coo_matrix(
        (
            np.ones(len(m2)),
            (
                m2[:, 0],
                m2[:, 1]
            )
        ),
        shape=(P, Q)
    )

    mask = M1 * M2

    match = mask.nonzero()

    match = list(
        zip(
            match[0],
            match[1]
        )
    )

    unmatched_O = tuple(
        set(range(O))
        -
        set(
            i
            for i, j in match
        )
    )

    unmatched_Q = tuple(
        set(range(Q))
        -
        set(
            j
            for i, j in match
        )
    )

    return (
        match,
        unmatched_O,
        unmatched_Q
    )


# ============================================================
# Indices -> Matches
# ============================================================

def _indices_to_matches(
    cost_matrix,
    indices,
    thresh
):

    matched_cost = cost_matrix[
        tuple(
            zip(*indices)
        )
    ]

    matched_mask = (
        matched_cost
        <= thresh
    )

    matches = indices[
        matched_mask
    ]

    unmatched_a = tuple(
        set(
            range(
                cost_matrix.shape[0]
            )
        )
        -
        set(
            matches[:, 0]
        )
    )

    unmatched_b = tuple(
        set(
            range(
                cost_matrix.shape[1]
            )
        )
        -
        set(
            matches[:, 1]
        )
    )

    return (
        matches,
        unmatched_a,
        unmatched_b
    )


# ============================================================
# Linear Assignment
# ============================================================

def linear_assignment(
    cost_matrix,
    thresh
):

    if cost_matrix.size == 0:

        return (
            np.empty(
                (0, 2),
                dtype=int
            ),
            tuple(
                range(
                    cost_matrix.shape[0]
                )
            ),
            tuple(
                range(
                    cost_matrix.shape[1]
                )
            )
        )

    matches = []

    cost, x, y = lap.lapjv(
        cost_matrix,
        extend_cost=True,
        cost_limit=thresh
    )

    for ix, mx in enumerate(x):

        if mx >= 0:

            matches.append(
                [
                    ix,
                    mx
                ]
            )

    unmatched_a = np.where(
        x < 0
    )[0]

    unmatched_b = np.where(
        y < 0
    )[0]

    matches = np.asarray(
        matches,
        dtype=int
    )

    if matches.size == 0:

        matches = np.empty(
            (0, 2),
            dtype=int
        )

    return (
        matches,
        unmatched_a,
        unmatched_b
    )


# ============================================================
# IoU
# ============================================================

def ious(
    atlbrs,
    btlbrs
):
    """
    Compute IoU matrix.

    atlbrs:
        list[tlbr] / ndarray

    btlbrs:
        list[tlbr] / ndarray

    return:
        IoU matrix
    """

    iou_matrix = np.zeros(
        (
            len(atlbrs),
            len(btlbrs)
        ),
        dtype=np.float64
    )

    if iou_matrix.size == 0:
        return iou_matrix

    iou_matrix = bbox_ious(

        np.ascontiguousarray(
            atlbrs,
            dtype=np.float64
        ),

        np.ascontiguousarray(
            btlbrs,
            dtype=np.float64
        )

    )

    return iou_matrix


# ============================================================
# IoU Distance
# ============================================================

def iou_distance(
    atracks,
    btracks
):

    if (
        (
            len(atracks) > 0
            and isinstance(
                atracks[0],
                np.ndarray
            )
        )
        or
        (
            len(btracks) > 0
            and isinstance(
                btracks[0],
                np.ndarray
            )
        )
    ):

        atlbrs = atracks
        btlbrs = btracks

    else:

        atlbrs = [
            track.tlbr
            for track in atracks
        ]

        btlbrs = [
            track.tlbr
            for track in btracks
        ]

    _ious = ious(
        atlbrs,
        btlbrs
    )

    cost_matrix = (
        1
        -
        _ious
    )

    return cost_matrix


# ============================================================
# Velocity IoU Distance
# ============================================================

def v_iou_distance(
    atracks,
    btracks
):

    if (
        (
            len(atracks) > 0
            and isinstance(
                atracks[0],
                np.ndarray
            )
        )
        or
        (
            len(btracks) > 0
            and isinstance(
                btracks[0],
                np.ndarray
            )
        )
    ):

        atlbrs = atracks
        btlbrs = btracks

    else:

        atlbrs = [
            track.tlwh_to_tlbr(
                track.pred_bbox
            )
            for track in atracks
        ]

        btlbrs = [
            track.tlwh_to_tlbr(
                track.pred_bbox
            )
            for track in btracks
        ]

    _ious = ious(
        atlbrs,
        btlbrs
    )

    cost_matrix = (
        1
        -
        _ious
    )

    return cost_matrix


# ============================================================
# Embedding Distance
# ============================================================

def embedding_distance(
    tracks,
    detections,
    metric="cosine"
):

    cost_matrix = np.zeros(
        (
            len(tracks),
            len(detections)
        ),
        dtype=np.float64
    )

    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray(
        [
            track.curr_feat
            for track in detections
        ],
        dtype=np.float64
    )

    track_features = np.asarray(
        [
            track.smooth_feat
            for track in tracks
        ],
        dtype=np.float64
    )

    cost_matrix = np.maximum(

        0.0,

        cdist(
            track_features,
            det_features,
            metric
        )

    )

    return cost_matrix


# ============================================================
# Gate Cost Matrix
# ============================================================

def gate_cost_matrix(
    kf,
    cost_matrix,
    tracks,
    detections,
    only_position=False
):

    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = (
        2
        if only_position
        else 4
    )

    gating_threshold = (
        kalman_filter.chi2inv95[
            gating_dim
        ]
    )

    measurements = np.asarray(
        [
            det.to_xyah()
            for det in detections
        ]
    )

    for row, track in enumerate(
        tracks
    ):

        gating_distance = (
            kf.gating_distance(

                track.mean,

                track.covariance,

                measurements,

                only_position

            )
        )

        cost_matrix[
            row,
            gating_distance
            >
            gating_threshold
        ] = np.inf

    return cost_matrix


# ============================================================
# Fuse Motion
# ============================================================

def fuse_motion(
    kf,
    cost_matrix,
    tracks,
    detections,
    only_position=False,
    lambda_=0.98
):

    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = (
        2
        if only_position
        else 4
    )

    gating_threshold = (
        kalman_filter.chi2inv95[
            gating_dim
        ]
    )

    measurements = np.asarray(
        [
            det.to_xyah()
            for det in detections
        ]
    )

    for row, track in enumerate(
        tracks
    ):

        gating_distance = (
            kf.gating_distance(

                track.mean,

                track.covariance,

                measurements,

                only_position,

                metric="maha"

            )
        )

        cost_matrix[
            row,
            gating_distance
            >
            gating_threshold
        ] = np.inf

        cost_matrix[row] = (

            lambda_
            *
            cost_matrix[row]

            +

            (
                1
                -
                lambda_
            )
            *
            gating_distance
        )

    return cost_matrix


# ============================================================
# Fuse IoU
# ============================================================

def fuse_iou(
    cost_matrix,
    tracks,
    detections
):

    if cost_matrix.size == 0:
        return cost_matrix

    reid_sim = (
        1
        -
        cost_matrix
    )

    iou_dist = iou_distance(
        tracks,
        detections
    )

    iou_sim = (
        1
        -
        iou_dist
    )

    fuse_sim = (
        reid_sim
        *
        (
            1
            +
            iou_sim
        )
        /
        2
    )

    det_scores = np.array(
        [
            det.score
            for det in detections
        ]
    )

    det_scores = np.expand_dims(
        det_scores,
        axis=0
    ).repeat(
        cost_matrix.shape[0],
        axis=0
    )

    fuse_cost = (
        1
        -
        fuse_sim
    )

    return fuse_cost


# ============================================================
# Fuse Detection Score
# ============================================================

def fuse_score(
    cost_matrix,
    detections
):

    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = (
        1
        -
        cost_matrix
    )

    det_scores = np.array(
        [
            det.score
            for det in detections
        ]
    )

    det_scores = np.expand_dims(
        det_scores,
        axis=0
    ).repeat(
        cost_matrix.shape[0],
        axis=0
    )

    fuse_sim = (
        iou_sim
        *
        det_scores
    )

    fuse_cost = (
        1
        -
        fuse_sim
    )

    return fuse_cost