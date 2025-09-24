# clustering_utils.py  (drop into your repo)
from pathlib import Path
import math
import numpy as np
import pandas as pd

# use your logger if available
try:
    from config import logger
    _log = logger.setup_logger()
    def _info(msg, *args): _log.info(msg, *args)
    def _warn(msg, *args): _log.warning(msg, *args)
    def _debug(msg, *args): _log.debug(msg, *args)
except Exception:
    def _info(msg, *args): print(msg % args if args else msg)
    def _warn(msg, *args): print("WARN:", msg % args if args else "WARN: " + msg)
    def _debug(msg, *args): pass

# circular mean for yaw in degrees (handles None)
def circular_mean_deg(angles_deg):
    """
    angles_deg: iterable of angles in degrees (may include None)
    returns None if no valid angles, else mean angle in degrees in [0,360)
    """
    vals = [a for a in angles_deg if a is not None]
    if not vals:
        return None
    rad = np.deg2rad(vals)
    s = np.sum(np.sin(rad))
    c = np.sum(np.cos(rad))
    if s == 0 and c == 0:
        return vals[0]  # degenerate: return first
    mean_rad = math.atan2(s, c)
    mean_deg = (math.degrees(mean_rad) + 360.0) % 360.0
    return float(mean_deg)

def cluster_detections_dbscan(all_detections, eps=0.5, min_samples=1, by_class=True,
                             metric='euclidean', agg_method='highest_conf'):
    """
    Cluster detections (list of dicts) using DBSCAN.

    Parameters
    ----------
    all_detections : list of dict
        Each detection dict must include 'mapped_x' and 'mapped_y' (floormap coords)
        and ideally 'class_name' and 'conf' (confidence).
    eps : float
        DBSCAN eps parameter (distance threshold). Units = same units as mapped_x/mapped_y (meters or feet).
    min_samples : int
        DBSCAN min_samples parameter. Set to 1 to allow singletons.
    by_class : bool
        If True, run DBSCAN separately per class (recommended). If False, cluster across all classes.
    metric : str or callable
        Passed to sklearn DBSCAN metric.
    agg_method : 'highest_conf' | 'centroid' | 'median'
        How to pick cluster representative:
          - 'highest_conf' : choose detection with highest 'conf' in cluster
          - 'centroid'     : average mapped_x/mapped_y (weighted by conf if present)
          - 'median'       : median location

    Returns
    -------
    clusters_df : pd.DataFrame
        One row per cluster with columns:
          ['cluster_id','class_name','centroid_x','centroid_y','rep_index','rep_conf',
           'rep_video_frame_index','rep_arkit_index','rep_bbox','rep_crop_path','members_count','member_indices','mean_yaw_deg']
    detection_to_cluster : dict
        mapping detection_list_index -> (class_name, cluster_id)
    """

    # validate input
    if not isinstance(all_detections, (list, tuple)):
        raise ValueError("all_detections must be a list (as returned by your detection function)")

    # convert to DataFrame for convenience, preserving original index
    df = pd.DataFrame(all_detections).reset_index().rename(columns={'index':'det_idx'})
    # require mapped_x,mapped_y
    if 'mapped_x' not in df.columns or 'mapped_y' not in df.columns:
        raise ValueError("each detection must include 'mapped_x' and 'mapped_y'")

    # try to import sklearn DBSCAN
    try:
        from sklearn.cluster import DBSCAN
    except Exception as e:
        raise RuntimeError("scikit-learn is required for DBSCAN clustering. Install with `pip install scikit-learn`. Error: " + str(e))

    clusters = []
    detection_to_cluster = {}

    next_global_cluster_id = 0

    if by_class:
        groups = df.groupby(df['class_name'].fillna('UNKNOWN'))
    else:
        groups = [('__ALL__', df)]

    for cls_name, subdf in groups:
        pts = subdf[['mapped_x', 'mapped_y']].to_numpy(dtype=float)
        if pts.shape[0] == 0:
            continue

        if pts.shape[0] == 1:
            labels = np.array([0])
        else:
            db = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric=metric)
            labels = db.fit_predict(pts)  # -1 for noise (if min_samples>1), else >=0 cluster labels

            # convert noise (-1) to unique singleton clusters if min_samples==1 or keep as -1
            if np.any(labels == -1):
                if min_samples <= 1:
                    # assign each -1 its own cluster id appended after current max label
                    max_lab = labels.max()
                    lab_map = {}
                    next_lab = max_lab + 1
                    for i, lab in enumerate(labels):
                        if lab == -1:
                            lab_map[i] = next_lab
                            next_lab += 1
                    for i, new_lab in lab_map.items():
                        labels[i] = new_lab
                # else keep -1 (noise); we'll ignore cluster -1 rows when creating clusters

        # remap local labels to global cluster ids to avoid collisions across classes
        unique_labels = sorted(set([l for l in labels if l != -1]))
        local_to_global = {}
        for l in unique_labels:
            local_to_global[l] = next_global_cluster_id
            next_global_cluster_id += 1

        for local_label in unique_labels:
            # select rows with this local label
            idxs = np.where(labels == local_label)[0]  # indices into subdf (0..n-1)
            member_rows = subdf.iloc[idxs]
            member_indices = member_rows['det_idx'].tolist()  # original list indices
            members_count = len(member_rows)

            # choose representative
            rep_row = None
            if agg_method == 'highest_conf' and 'conf' in member_rows.columns:
                # pick highest confidence
                rep_idx = member_rows['conf'].astype(float).idxmax()
                rep_row = member_rows.loc[rep_idx]
            elif agg_method == 'median':
                median_x = float(np.median(member_rows['mapped_x'].astype(float)))
                median_y = float(np.median(member_rows['mapped_y'].astype(float)))
                # find row closest to median
                d2 = (member_rows['mapped_x'].astype(float) - median_x)**2 + (member_rows['mapped_y'].astype(float) - median_y)**2
                rep_idx = d2.idxmin()
                rep_row = member_rows.loc[rep_idx]
            else:  # centroid (or fallback)
                # weighted centroid if conf exists
                xs = member_rows['mapped_x'].astype(float).to_numpy()
                ys = member_rows['mapped_y'].astype(float).to_numpy()
                if 'conf' in member_rows.columns:
                    ws = member_rows['conf'].astype(float).to_numpy()
                    if np.sum(ws) > 0:
                        cx = float(np.sum(xs * ws) / np.sum(ws))
                        cy = float(np.sum(ys * ws) / np.sum(ws))
                    else:
                        cx = float(np.mean(xs)); cy = float(np.mean(ys))
                else:
                    cx = float(np.mean(xs)); cy = float(np.mean(ys))
                # find row nearest to centroid as representative
                d2 = (member_rows['mapped_x'].astype(float) - cx)**2 + (member_rows['mapped_y'].astype(float) - cy)**2
                rep_idx = d2.idxmin()
                rep_row = member_rows.loc[rep_idx]

            # compute centroid/summary for cluster
            centroid_x = float(np.mean(member_rows['mapped_x'].astype(float)))
            centroid_y = float(np.mean(member_rows['mapped_y'].astype(float)))
            mean_yaw = circular_mean_deg(member_rows.get('object_yaw_deg', pd.Series([None]*len(member_rows))).tolist())

            global_cid = local_to_global[local_label]

            # build cluster record
            rec = {
                'cluster_id': int(global_cid),
                'class_name': cls_name,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'rep_index': int(rep_row['det_idx']),
                'rep_conf': float(rep_row['conf']) if 'conf' in rep_row and not pd.isna(rep_row['conf']) else None,
                'rep_video_frame_index': int(rep_row['video_frame_index']) if 'video_frame_index' in rep_row else None,
                'rep_arkit_index': int(rep_row['arkit_index']) if 'arkit_index' in rep_row else None,
                'rep_bbox': rep_row.get('bbox', None),
                'rep_crop_path': rep_row.get('crop_path', None),
                'members_count': int(members_count),
                'member_indices': member_indices,
                'mean_yaw_deg': mean_yaw
            }
            clusters.append(rec)

            # fill detection_to_cluster mapping
            for det_list_idx in member_indices:
                detection_to_cluster[int(det_list_idx)] = (cls_name, int(global_cid))

    clusters_df = pd.DataFrame(clusters)
    if clusters_df.empty:
        _info("cluster_detections_dbscan: no clusters found (empty input).")
    else:
        _info("cluster_detections_dbscan: produced %d clusters", clusters_df.shape[0])
    return clusters_df, detection_to_cluster
