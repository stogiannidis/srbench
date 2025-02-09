#!/usr/bin/env python3
import os
import random
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec

# Global dictionary of polycube shapes.
SHAPES = {
    "L3D": [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1)],
    "Stair": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 1, 1), (3, 1, 1)],
    "Cross3D": [
        (1, 1, 1),
        (0, 1, 1),
        (2, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 1, 0),
        (1, 1, 2),
    ],
    "Ring": [  # A ring-like shape.
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
    ],
    "Snake": [  # A snake-like polycube, turning in three dimensions.
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 2, 2),
        (2, 3, 2),
    ],
    "TShape": [  # T‐shaped polycube with a central protrusion.
        (1, 0, 0),  # bottom center
        (0, 1, 0),  # left arm
        (1, 1, 0),  # center junction
        (2, 1, 0),  # right arm
        (1, 2, 0),  # top arm
        (1, 1, 1),  # cube on top of the center
    ],
    "Spiral": [  # A spiral staircase polycube, twisting upward.
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (0, 1, 1),
        (0, 2, 1),
        (0, 2, 2),
        (-1, 2, 2),
    ],
    "Branch": [  # A branching polycube with an offshoot.
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (1, 1, 0),
        (2, 1, 0),
        (2, 1, 1),
    ],
    "DoubleSnake": [  # A double-layer snake polycube weaving between two layers.
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 1),
        (1, 1, 0),
        (2, 1, 0),
        (2, 1, 1),
        (2, 2, 1),
        (2, 2, 0),
    ],
    "Zigzag": [  # A zigzag polycube with multi‐axis turns.
        (0, 0, 0),
        (1, 0, 0),  # right
        (1, 1, 0),  # forward
        (2, 1, 0),  # right again
        (2, 1, 1),  # upward turn
        (2, 2, 1),  # forward
        (3, 2, 1),  # right
        (3, 2, 2),  # upward finish
    ],
}

# Mapping of visually similar shapes based on their coordinates.
# Each key maps to a list of other shape names considered visually similar.
SIMILAR_MAPPING = {
    "L3D": ["TShape", "Stair"],
    "Stair": ["TShape", "L3D"],
    "Cross3D": ["Ring"],
    "Ring": ["Cross3D"],
    "Snake": ["Spiral", "Zigzag"],
    "TShape": ["L3D", "Stair"],
    "Spiral": ["Snake", "Zigzag"],
    "Branch": ["DoubleSnake"],
    "DoubleSnake": ["Branch"],
    "Zigzag": ["Snake", "Spiral"],
}


def set_axes_equal(ax, all_vertices):
    """
    Make the aspect ratio of the 3D plot equal so our shape is not distorted.
    """
    all_vertices = np.array(all_vertices)
    x_limits = [np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])]
    y_limits = [np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])]
    z_limits = [np.min(all_vertices[:, 2]), np.max(all_vertices[:, 2])]

    # Compute ranges and midpoints.
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


def cube_vertices(origin, size=1.0):
    """
    Given an (x, y, z) origin and a cube edge size, return the 8 corner vertices of that cube.
    """
    x, y, z = origin
    return np.array(
        [
            [x, y, z],
            [x + size, y, z],
            [x + size, y + size, z],
            [x, y + size, z],
            [x, y, z + size],
            [x + size, y, z + size],
            [x + size, y + size, z + size],
            [x, y + size, z + size],
        ]
    )


def plot_cubes(ax, vertices, title=""):
    """
    Plot each cube face-by-face using its 8 vertices,
    and optionally set a subplot title.
    """
    n_cubes = len(vertices) // 8
    vertices_reshaped = vertices.reshape((n_cubes, 8, 3))
    for cube_verts in vertices_reshaped:
        faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # side
            [2, 3, 7, 6],  # side
            [1, 2, 6, 5],  # front
            [0, 3, 7, 4],  # back
        ]
        for face in faces:
            polygon = Poly3DCollection(
                [cube_verts[face]],
                facecolors="white",
                edgecolors="black",
                alpha=1.0,
            )
            ax.add_collection3d(polygon)
    ax.set_title(title, fontsize=12)
    set_axes_equal(ax, vertices)


def generate_shape_vertices(shape_name, cube_size=1.0):
    """
    Return an array of *all* corner vertices for the given shape name.
    """
    if shape_name not in SHAPES:
        raise ValueError(f"Unknown shape {shape_name}")
    cube_origins = SHAPES[shape_name]
    all_vertices = []
    for origin in cube_origins:
        corners = cube_vertices(origin, size=cube_size)
        all_vertices.append(corners)
    return np.vstack(all_vertices)  # Nx3 array


def get_transformed_candidate(transformation_func, original, max_attempts=10):
    """
    Apply the transformation function to the original vertices until the
    candidate differs from the original (within a tolerance).
    """
    for _ in range(max_attempts):
        candidate = transformation_func(original)
        if not np.allclose(candidate, original, atol=1e-6):
            return candidate
    return candidate


def transform_rotate(vertices):
    """
    Rotate the shape randomly and return the new vertex array.
    (This is our correct candidate transformation.)
    """
    angle_x = np.deg2rad(np.random.uniform(10, 45))
    angle_y = np.deg2rad(np.random.uniform(10, 45))
    angle_z = np.deg2rad(np.random.uniform(10, 45))
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )
    R = Rz @ Ry @ Rx
    center = vertices.mean(axis=0)
    shifted = vertices - center
    rotated = (R @ shifted.T).T
    return rotated + center


def transform_scale(vertices):
    """
    Apply a slight, non-uniform scaling to the shape.
    Scale factors are chosen very close to 1.0 for a subtle effect.
    """
    scale_factors = np.array(
        [
            np.random.uniform(0.9, 1.1),
            np.random.uniform(0.9, 1.1),
            np.random.uniform(0.9, 1.1),
        ]
    )
    center = vertices.mean(axis=0)
    shifted = vertices - center
    scaled = shifted * scale_factors
    return scaled + center


def transform_shear(vertices):
    """
    Apply a slight shear transformation to the shape.
    The shear factors are small for a more subtle effect.
    """
    shear_factor1 = np.random.uniform(0.05, 0.15)
    shear_factor2 = np.random.uniform(0.05, 0.15)
    shear_matrix = np.array([[1, shear_factor1, 0], [shear_factor2, 1, 0], [0, 0, 1]])
    center = vertices.mean(axis=0)
    shifted = vertices - center
    sheared = (shear_matrix @ shifted.T).T
    return sheared + center


def transform_mirror(vertices):
    """
    Mirror (reflect) the shape along a randomly chosen axis.
    """
    center = vertices.mean(axis=0)
    shifted = vertices - center
    mirrored = shifted.copy()
    axis = random.choice([0, 1, 2])
    mirrored[:, axis] = -mirrored[:, axis]
    return mirrored + center


def transform_rot_mix(vertices):
    """
    Combine a subtle rotation with a shear transformation.
    First, apply a small rotation (5-15 degrees around the z-axis),
    then apply a shear transformation with increased shear factors so that the shear effect is more visible.
    """
    # Apply a small rotation around the z-axis.
    angle = np.deg2rad(np.random.uniform(5, 15))
    Rz = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    center = vertices.mean(axis=0)
    shifted = vertices - center
    rotated = (Rz @ shifted.T).T + center

    # Now apply a shear transformation with increased shear factors.
    shear_factor1 = np.random.uniform(
        0.1, 0.2
    )  # Increased range for a more visible shear effect.
    shear_factor2 = np.random.uniform(0.1, 0.2)
    shear_matrix = np.array([[1, shear_factor1, 0], [shear_factor2, 1, 0], [0, 0, 1]])
    shifted = rotated - center
    sheared = (shear_matrix @ shifted.T).T + center
    return sheared


def get_visually_similar_candidate(chosen_shape_name, original_vertices, cube_size=1.0):
    """
    Return a candidate from SHAPES that is visually similar to the chosen shape,
    using a pre-defined mapping based on coordinates.
    The candidate is generated from a shape listed in SIMILAR_MAPPING (if available)
    and then a random rotation is applied.

    To avoid errors when comparing arrays of different shapes, we first check if
    the candidate's shape is different from the original; if so, we return it immediately.
    Otherwise, we check numerical closeness.
    """
    if chosen_shape_name in SIMILAR_MAPPING:
        similar_candidates = SIMILAR_MAPPING[chosen_shape_name]
        if similar_candidates:
            random.shuffle(similar_candidates)
            for similar_shape_name in similar_candidates:
                candidate_vertices = generate_shape_vertices(
                    similar_shape_name, cube_size
                )
                candidate_vertices = get_transformed_candidate(
                    transform_rotate, candidate_vertices
                )
                if (
                    candidate_vertices.shape != original_vertices.shape
                    or not np.allclose(candidate_vertices, original_vertices, atol=1e-6)
                ):
                    return ("visually_similar", candidate_vertices)
    # Fallback: if no mapping exists or no candidate is found, return None.
    return None


def generate_one_image(index):
    """
    Generate a single image with a complex polycube shape and four candidate transformations.
    The correct candidate (rotate) is always generated, while three wrong candidates are produced:
      - Two wrong candidates are produced by randomly chosen transformation functions.
      - One wrong candidate is a visually similar shape (if available) from the mapping.
    The candidate order is shuffled so that the correct candidate appears in a random position.
    Metadata about the image is appended to metadata.jsonl.
    """
    # Choose an original shape.
    shapes_list = list(SHAPES.keys())
    shape_name = random.choice(shapes_list)
    original_vertices = generate_shape_vertices(shape_name, cube_size=1.0)

    # Correct candidate (pure rotation) ensuring it differs from the original.
    rotate_vertices = get_transformed_candidate(transform_rotate, original_vertices)

    # Define a pool of wrong transformation functions.
    wrong_transformations = {
        "scale": transform_scale,
        "shear": transform_shear,
        "mirror": transform_mirror,
        "rot_mix": transform_rot_mix,  # Combination: small rotation + more visible shear.
    }

    wrong_candidates = []
    # First, try to get a visually similar candidate from the mapping.
    vis_candidate = get_visually_similar_candidate(
        shape_name, original_vertices, cube_size=1.0
    )
    if vis_candidate is not None:
        wrong_candidates.append(vis_candidate)
    # Now, choose enough wrong transformation candidates to reach 3 wrong candidates.
    num_wrong_needed = 3 - len(wrong_candidates)
    wrong_keys = random.sample(list(wrong_transformations.keys()), num_wrong_needed)
    for key in wrong_keys:
        candidate = get_transformed_candidate(
            wrong_transformations[key], original_vertices
        )
        wrong_candidates.append((key, candidate))

    # Build the full candidate list (1 correct + 3 wrong = 4 total) and shuffle.
    candidates = [("rotate", rotate_vertices)] + wrong_candidates
    random.shuffle(candidates)

    # Determine the correct candidate index (0-based).
    correct_candidate_index = [
        i for i, cand in enumerate(candidates) if cand[0] == "rotate"
    ][0]

    # Create the figure with subplots.
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 4, height_ratios=[1, 1], wspace=0.3, hspace=0.3)

    # Top row: original shape.
    ax_orig = fig.add_subplot(gs[0, :], projection="3d")
    plot_cubes(ax_orig, original_vertices, title=f"Original Shape ({shape_name})")

    # Bottom row: 4 candidate subplots.
    for i in range(4):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        label, candidate_vertices = candidates[i]
        plot_cubes(ax, candidate_vertices, title=f"Candidate {i + 1}")

    # Save the figure.
    filename = f"{shape_name}_mrt_{index}.png"
    outdir = "mrt_images"
    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Write metadata to metadata.jsonl.
    metadata = {
        "filename": filename,
        "candidate_order": [label for label, _ in candidates],
        "correct_candidate": correct_candidate_index + 1,  # 1-indexed
    }
    with open("mrt_images/metadata.jsonl", "a") as f:
        f.write(json.dumps(metadata) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple mental-rotation test images of complex polycubes."
    )
    parser.add_argument(
        "--num_images", "-n", type=int, default=1, help="Number of images to generate."
    )
    args = parser.parse_args()

    for i in range(args.num_images):
        generate_one_image(i)


if __name__ == "__main__":
    main()
