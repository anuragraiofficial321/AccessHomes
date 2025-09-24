# 1) High-level overview (one-sentence per stage)

1. **Read ARKit data** (`arkitData.json`) and extract camera poses (3D position + rotation matrix) for each frame.
2. **Project camera positions** from 3D → 2D using a chosen axis projection (e.g. `x,-z`) to get a bird’s-eye trajectory.
3. **Load floorplan polygons** (rooms / walls / fixed furniture) from `floor.json`.
4. **Find similarity transform** (scale, rotation, translation) that maps the projected camera path onto the floorplan coordinate system — either using control points (Umeyama) or an automatic heuristic that tries multiple projections and chooses the best alignment.
5. **Interpolate / fill missing positions** to create smooth tracked path.
6. **Run object detector** (YOLO) over a video to find first-seen detections by class, record bounding boxes and per-frame info.
7. **(Optional) Depth estimation** for detected crops (Zoe or another depth model) to estimate distance along camera ray.
8. **Back-project bounding box center + depth into world 3D** using intrinsics + camera rotation to get object 3D world position.
9. **Map that 3D position to floorplan 2D coordinates** using the previously computed similarity transform.
10. **Plot** detections and floor polygons; optionally annotate object yaw (orientation), save CSV / images.

---

# 2) Data & coordinate systems — what you actually have

* **ARKit pose** (per frame) — often stored as a 4×4 homogeneous matrix `T_cam` or as a 12/16-element array. When parsed you get:

  * `pos` = camera position in ARKit/world coords (3×1): `[x, y, z]`
  * `rot` = 3×3 camera rotation matrix (R\_cam) transforming camera vectors to world coordinates (or vice versa depending on how ARKit exports it).

* **Floorplan**:

  * Polygons representing rooms in some 2D coordinate system (units: meters or feet).
  * Optional `compassHeading` orienting the plan relative to world north.

* **Video frames & intrinsics**:

  * Image width/height; camera intrinsics `(fx, fy, cx, cy)` sometimes found in metadata — focal lengths and principal point in pixels.

* **Detections**:

  * Bounding boxes `(x1,y1,x2,y2)` in image pixel coordinates plus detection confidence and class.

---

# 3) Key math & algorithms (with equations)

I’ll concentrate on the *important physics / geometry* bits.

## 3.1 Parsing a homogeneous matrix → camera position and rotation

A 4×4 homogeneous transform `T_cam` typically looks like:

$$
T =
\begin{bmatrix}
R_{3\times3} & t_{3\times1} \\
0 & 1
\end{bmatrix}
$$

* `R` is the rotation matrix.
* `t` is the translation vector.
  Your `parse_homogeneous_matrix` extracts `pos = t` and `rot = R`.
  Note: sometimes ARKit stores `T` as camera→world or world→camera. You detect that via heuristics or by checking shapes/orientations.

## 3.2 Projecting 3D → 2D (bird’s-eye)

You choose a projection string like `"x,-z"` or `"x,z"`. That means:

If 3D point is `p = [x, y, z]`, then:

* `proj="x,-z"` → `p2 = (x, -z)`
* `proj="x,z"` → `p2 = (x, z)`
* `proj="y,-z"` → `p2 = (y, -z)`

This is a **simple orthographic projection** (no perspective), used to get floorplan coordinates from camera *positions* rather than from observed objects. This is good because camera positions are usually near the floor plane and easier to align.

## 3.3 Umeyama similarity (compute s, R, t to align two 2D point sets)

The Umeyama algorithm computes a similarity transform that minimizes squared error between source points `X` and destination points `Y`:

Find scale `s`, rotation matrix `R` (2×2) and translation `t` (2×1) such that:

$$
Y \approx s R X + t
$$

Umeyama closed-form steps (summary):

1. Compute centroids: $\mu_X, \mu_Y$.
2. Compute centered covariance: $\Sigma = \frac{1}{n}\sum (X_i - \mu_X)(Y_i - \mu_Y)^T$.
3. SVD: $\Sigma = U \Lambda V^T$.
4. Rotation: $R = U \operatorname{diag}(1, \det(UV^T)) V^T$.
5. Scale: $s = \frac{1}{\sigma_X^2} \operatorname{trace}(\Lambda \cdot \operatorname{diag}(1, \det(UV^T)))$.
6. Translation: $t = \mu_Y - s R \mu_X$.

Umeyama is robust and provides scale — useful when ARKit units (meters) differ from floorplan units (feet) or there is a scale uncertainty.

## 3.4 Auto-mapping heuristic

If control points are not provided, the script:

1. Tries multiple **projection options** (e.g. `x,-z`, `x,z`, `-x,-z`, `-x,z`, `y,-z`).
2. For each projection:

   * Project camera positions into 2D (`p2`).
   * Use an optimization/heuristic (`auto_map_and_choose`) that tries rotating & scaling `p2` to fit into floor polygon envelope and computes a **score**: how many projected camera points fall inside room polygons, or overlap with the plan.
3. Choose the projection with the **highest score**. Then compute an Umeyama similarity between `p2` and the mapped candidate to finalize `s_map`, `R_map`, `t_map`.

This is effective because the camera path should lie inside rooms in the floorplan.

## 3.5 Interpolation of missing frames

`interp_missing` fills gaps in the projected path (frames where ARKit didn't supply a pose). Typical methods:

* Linear interpolation for x,y coordinates across missing frames.
* Optionally smoothing with moving-average or splines.

This avoids plotting holes and gives stable camera path used for mapping.

## 3.6 Orientation voting (R vs R.T and z forward sign)

Camera rotation matrices sometimes come as transposed depending on export convention. The script:

* For every frame with a rotation matrix, compare `R` vs `R.T` (transpose) and compute which one better matches motion between neighbor frames (the camera forward direction should align with direction of consecutive positions).
* Use `choose_best_rotation_and_sign` to compute for a frame whether to use `R` or `R.T`, and whether the forward z axis should be `+z` or `-z`.
* Globally vote across frames and apply majority choice. This avoids per-frame inconsistent rotations.

## 3.7 Backprojecting image point + depth → 3D world (the heart of object-to-floor mapping)

Given:

* Image pixel coordinates of object center: `(u,v)` in pixels.
* Camera intrinsics: `fx, fy, cx, cy`.
* Camera rotation `R_cam` (3×3) and camera position `C` in world coordinates.
* Estimated depth along camera forward direction: `depth_val` (units match camera/world units).
* `z_sign`: whether camera's camera-space forward axis corresponds to `+z` or `-z`.

**Steps**:

1. Compute normalized camera ray direction in camera coordinates:

$$
d_{cam} = \begin{bmatrix} (u - c_x) / f_x \\ (v - c_y) / f_y \\ z\_sign \end{bmatrix}
$$

The code normalizes this direction:

$$
d_{cam\_n} = \frac{d_{cam}}{\| d_{cam} \|}.
$$

2. Rotate to world coordinates:

$$
d_{world} = R \cdot d_{cam\_n}
$$

(If `R` maps camera→world; if `R` is world→camera you would use `R^T` accordingly — that's why orientation voting is important.)

3. Normalize direction and compute 3D object location:

$$
p_{obj\_world} = C + depth\_val \cdot \frac{d_{world}}{\|d_{world}\|}
$$

This places the object along the camera ray at the estimated distance.

4. Project that 3D object into 2D floorplan coordinates using the previously chosen **projection** (e.g. take x and -z) and then apply similarity transform:

$$
p_{obj2D} = s\_map R\_map p_{proj} + t\_map
$$

This yields the location on the floorplan where the object is plotted.

**Important note**: depth must be in same units as `C`. If depth comes from a model that produces relative units, convert to meters/feet.

## 3.8 Compute yaw / object orientation

From the world direction vector `d_world` (the camera→object direction), the yaw is computed as:

$$
\text{yaw\_deg} = \operatorname{deg}(\operatorname{atan2}(d_x, -d_z))
$$

This chooses a yaw angle in degrees where 0° points aligned with a floor +z axis convention used in plotting. The exact formula depends on your chosen floor axis convention (x vs z).

---

# 4) Worked numeric example for `compute_object_world_and_mapped`

Suppose:

* Camera position `C = [2.0, 1.6, 3.0]` (meters).
* Camera rotation `R = I` (identity — camera axes ≈ world axes for simplicity).
* Intrinsics: image `640×480`, `fx = fy = 800`, `cx = 320`, `cy = 240`.
* Bounding box center in image: `u = 400`, `v = 250`.
* Depth estimate along forward direction: `depth_val = 2.5` meters.
* `z_sign = 1.0`.
* Projection: `proj="x,-z"`.
* similarity `s_map = 1.2`, `R_map = [[cosθ,-sinθ],[sinθ,cosθ]]` with θ=10°, `t_map = [1.0, 0.5]` for example.

Step-by-step:

1. Pixel to camera direction:

$$
d_{cam} = \left[\frac{400-320}{800},\frac{250-240}{800},1.0\right] = [0.1,\,0.0125,\,1.0]
$$

Norm:

$$
\|d_{cam}\| = \sqrt{0.1^2 + 0.0125^2 + 1^2} \approx \sqrt{1.01015625} \approx 1.00507
$$

Normalized direction:

$$
d_{cam\_n} \approx [0.0995,\,0.0124,\,0.99497]
$$

2. Rotate to world (`R = I`), so `d_world = d_cam_n`.

3. Compute world position:

$$
p_{obj\_world} = C + depth\_val \cdot d_{world} \approx [2.0,1.6,3.0] + 2.5 \cdot [0.0995,0.0124,0.99497] \\
\approx [2.0+0.2488,\,1.6+0.031,\,3.0+2.4874] = [2.2488,\,1.631,\,5.4874]
$$

4. Project to floorplan using `x,-z`:

$$
p\_proj = (x, -z) = (2.2488, -5.4874)
$$

5. Apply similarity: rotate by θ=10°, scale 1.2, translate \[1.0, 0.5]. If `R_map` rotation matrix with θ=10° ≈ \[\[0.9848, -0.1736],\[0.1736,0.9848]]:

$$
p_{mapped} = 1.2 \cdot R\_map \cdot p\_proj + t\_map
$$

(You can numerically compute this — the result is the 2D location plotted on the floorplan.)

This shows how a pixel center + depth becomes a location on the floor plan.

---

# 5) Why normalize `d_cam` before rotating?

Normalization removes the arbitrary scale introduced by the `z_sign` placeholder (we use a unit camera direction) — then multiply by `depth_val` to place object at the real distance. If you didn’t normalize, the magnitude of `(u-cx)/fx` would incorrectly scale distance.

---

# 6) Depth sources & caveats

* **Zoe / monocular depth models**: produce *relative depth* or per-pixel depth maps. They may be in arbitrary units. You must convert model depth to real-world length. Strategies:

  * Use known-size objects or the camera intrinsics & geometry to convert to metric units.
  * Use stereo or SLAM data for scale — ARKit camera path gives scale in meters often, so you can calibrate depth by matching some known points (e.g., floor plane).
  * Use simple pinhole geometry if you have real-world object size or use bounding-box-based heuristics.

* **Depth accuracy**: monocular depth can have scale ambiguity and errors; best used to estimate relative depth or when combined with other cues.

---

# 7) YOLO detection integration (how first-seen works)

* Loop through frames in the video.
* For each frame, pass ROI or full frame to `detector.predict()` to obtain boxes, confidences, class ids.
* Maintain a dictionary `found_first` that maps class → first detection info (frame index, bounding box, confidence).
* When a new class is seen, compute its mapped floor location using the procedure in section 3.7, save crop image, annotated frame and stats to CSV.

This produces one representative point per class (“first-seen”) on the floorplan.

---

# 8) Plotting

`plot_floorplan()` draws:

* Room polygons (from `spaces`).
* Fixed furniture polygons or points.
* Camera path.
* Object markers (emojis or icons) at the mapped 2D coordinates.
* Optionally draw arrow showing object yaw.

Plotting is done in matplotlib (or similar), with axes set to floor bounds. Important to ensure axis aspect is `equal` to preserve geometry.

---

# 9) Typical failure modes and debugging tips

1. **Bad rotation signs** → objects plotted behind walls or flipped. Fix: use orientation voting, check `R` vs `R.T`. Log or visualize camera forward vectors vs motion vectors.
2. **Scale mismatch (meters vs feet)** → objects plotted too large or small. Fix: ensure `M_TO_FT` and `CONVERT_M_TO_FT` flags are correct.
3. **No intrinsics** → script uses a fallback (fx=fy=0.8\*max(w,h)) — this can be rough. Better: extract real intrinsics from video or ARKit metadata.
4. **Depth units mismatch** → depth must be same units as camera positions.
5. **Floorplan coordinate orientation** → compass heading or rotated plans lead to bad alignments. The auto mapping tries rotations, but check `compassHeading` in floormap.
6. **ARKit missing frames** → interpolation fills but can misalign at edges; ensure interpolation is reasonable.

---

# 10) Suggestions for improvements (practical)

* **Estimate depth scale** by using floor plane intersection: detect floor in depth map and compare camera height to known camera height to calibrate.
* **RANSAC for mapping**: when matching `p2`→floor features, use RANSAC to reject outliers before Umeyama.
* **Use ICP** (Iterative Closest Point) to refine mapping between projected camera path and floorplan geometry.
* **Temporal smoothing** for object placement (Kalman filter) to reduce jitter across frames.
* **Visual diagnostics**: make outputs like `camera_vectors.png` showing per-frame camera forward vectors overlaid on mapped path (helps debug orientation).
* **Confidence propagation**: store per-object uncertainty (depth variance) and visualize as error ellipses on the plan.

---

# 11) Small experiment snippets you can run (purely illustrative)

### 11.1 Visualize camera ray and final mapped point (quick test)

```python
import numpy as np
import math

# example numbers from the worked example
C = np.array([2.0, 1.6, 3.0])
fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
u, v = 400.0, 250.0
depth_val = 2.5
z_sign = 1.0
R = np.eye(3)

d_cam = np.array([(u - cx)/fx, (v - cy)/fy, z_sign])
d_cam_n = d_cam / np.linalg.norm(d_cam)
d_world = R @ d_cam_n
p_obj_world = C + depth_val * (d_world / np.linalg.norm(d_world))
print("object world:", p_obj_world)

# project to floor 2D x,-z
p_proj = np.array([p_obj_world[0], -p_obj_world[2]])
print("proj2D:", p_proj)
```

### 11.2 Example: compute yaw from direction

```python
d = d_world / np.linalg.norm(d_world)
dx, dz = d[0], d[2]
yaw_deg = math.degrees(math.atan2(dx, -dz)) % 360.0
print("yaw_deg:", yaw_deg)
```

---

# 12) ASCII flowchart (visual diagram)

```
[ READ ARKIT JSON ]    [ LOAD FLOORPLAN JSON ]    [ OPEN VIDEO ]
        |                        |                      |
        v                        v                      v
 [ PARSE POSES (pos, R) ] ----> [ project 3D -> 2D ]   [ run YOLO per-frame ]
        |                         |                     |
        |                         v                     v
        |                 [ auto_map_or_controlpts ] <- detect boxes
        |                         |                     |
        v                         v                     |
[ interp_missing -> proj2_all ] [ compute s,R,t (Umeyama) ] 
        |                         |                     |
        |                         v                     |
        +--------------------> [ mapped_all (camera path) ] 
                                  |
                                  v
                             [ for each detection ]
                                  |
                                  v
                 [ estimate depth (Zoe) / use heuristics ]
                                  |
                                  v
                 [ backproject pixel+depth -> p_obj_world ]
                                  |
                                  v
                 [ project p_obj_world -> floor2D + apply s,R,t ]
                                  |
                                  v
                 [ plot marker + yaw on floorplan; save CSV ]
```

---

# 13) Summary checklist (what to verify in your dataset to ensure pipeline works)

* ✅ ARKit poses are present and have 12/16 element matrices or `cameraTransform`.
* ✅ Floorplan `spaces` polygons exist and are in a continuous coordinate system.
* ✅ If using monocular depth, choose a method to convert model depth to real-world units.
* ✅ Intrinsics (`fx,fy,cx,cy`) either extracted or acceptable fallback provided.
* ✅ `CONTROL_POINTS` if available will lock mapping and produce accurate alignment.
* ✅ Logging (your new `logger`) will capture failed steps; inspect logs for `exception` messages.

---

# 14) Quick reference of important functions in your code and their responsibilities

* `extract_positions_list()` — read ARKit JSON and produce `positions3d` and `meta` (rotations + raw).
* `project_3d_to_2d(positions3d, proj)` — orthographic projection of camera positions.
* `umeyama_2d(src, dst)` — compute similarity transform `s, R, t`.
* `auto_map_and_choose(...)` — try projections & choose best mapping (score).
* `interp_missing(...)` — fill missing frames for plotting and mapping.
* `choose_best_rotation_and_sign(...)` — per-frame orientation correction; used for global voting.
* `YoloDetector` — object detector; returns boxes/conf/class.
* `init_zoe, get_zoe_depth_map` — optional monocular depth model.
* `compute_object_world_and_mapped()` — core: pixel+depth+intrinsics+R,C -> p\_obj\_world -> apply similarity -> mapped 2D and yaw.
* `plot_floorplan(...)` — draw plan, camera path, detections, emojis/icons.

---

# 15) Limitations, expected accuracy & best-case scenarios

* **Best case**: ARKit poses are accurate and in meters, camera intrinsics are known, objects are upright and not occluded, depth model has correct scale — resulting floor positions accurate to \~0.1–0.5 m (depending on depth model and camera).
* **Worst case**: monocular depth with wrong scale or missing intrinsics can give meter-scale errors; orientation mistakes can place objects behind walls.

---
