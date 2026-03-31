# DATASET GENERATION

This folder contains scripts used for COLMAP -> Objectron-style dataset generation.

## Core entry points
- `generate+objectron_dataset_colmap.py`: generate overlays + `annotations.json` from COLMAP outputs.
- `package_objectron_dataset.py`: package images/annotations/metadata with train/val/test split.
- `verify_objectron_overlays.py`: quick overlay sanity check.

## Mask + cleaning helpers
- `run_sam2_video_mask.py`
- `tools_sam2_segment.py`
- `run_masked_filter.py`

## Cuboid fitting/refinement helpers
- `fit_objectron_cuboid.py`
- `fit_objectron_cuboid_cylinder.py`
- `fit_objectron_cuboid_opt.py`
- `stabilize_cylinder_object_frame.py`
- `calibrate_cuboid_from_2d.py`
- `calibrate_cuboid_with_masks.py`
- `joint_optimize_cuboid_with_masks.py`
- `optimize_cuboid_silhouette_iou.py`
- `pareto_refine_cuboid.py`
- `refine_cuboid_fullpose_with_masks.py`
- `refine_cuboid_with_mask_edges.py`
- `refine_rotation_with_mask_edges.py`
- `refine_translation_with_masks.py`

## Visualization/debug
- `overlay_cuboid_colmap.py`
- `render_colmap_stride_views.py`
- `strides_viewer_colmap.py`
- `project_objectron_keypoints.py`
- `npz_viewer.py`
- `anchor_yaw_from_density.py`
