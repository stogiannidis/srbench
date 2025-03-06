#!/usr/bin/env python3
import os
import random
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # For drawing rectangles
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec

# Global dictionary of complex polycube shapes.
# Only complex, snake-like and related shapes are included.
SHAPES = {
    "Snake": [  # A basic snake-like polycube.
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 2, 2),
        (2, 3, 2),
    ],
    "Zigzag": [  # A zigzag polycube.
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (2, 1, 0),
        (2, 1, 1),
        (2, 2, 1),
        (3, 2, 1),
        (3, 2, 2),
    ],
    "SnakeComplex1": [  # A more complex, winding snake.
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 0),
        (2, 1, 1),
        (2, 2, 1),
        (1, 2, 1),
        (1, 3, 1),
        (1, 3, 2),
    ],
    "HookedCorner": [  # Formerly shapeB: a horizontal bar that hooks upward.
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
    ],
    "TopPlate": [  # Formerly shapeC: a vertical column topped with a horizontal plate.
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (0, 2, 1),
        (1, 2, 1),
        (2, 2, 1),
    ],
    "CornerStaircase": [  # Formerly shapeD: an L-shaped staircase.
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (3, 2, 0),
        (3, 3, 0),
    ],
    "TripleArm": [  # Formerly shapeG: a shape with three distinct arms.
        (3, -1, 0),
        (3, -1, 1),
        (3, -1, 2),
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
    ],
}

# Dynamic similar-object mapping: every shape is considered similar to every other shape.
all_shape_keys = list(SHAPES.keys())
SIMILAR_MAPPING = {
    key: [s for s in all_shape_keys if s != key] for key in all_shape_keys
}

OUTDIR = "data/mrt/complex"


def set_axes_equal(ax, all_vertices):
    """
    Make the aspect ratio of the 3D plot equal so our shape is not distorted.
    Remove the 3D cartesian axes by hiding panes, grid lines, and axis lines.
    """
    all_vertices = np.array(all_vertices)
    x_limits = [np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])]
    y_limits = [np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])]
    z_limits = [np.min(all_vertices[:, 2]), np.max(all_vertices[:, 2])]
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
    # Set a few ticks for reference.
    ax.set_xticks(np.linspace(x_limits[0], x_limits[1], 5))
    ax.set_yticks(np.linspace(y_limits[0], y_limits[1], 5))
    ax.set_zticks(np.linspace(z_limits[0], z_limits[1], 5))
    # Remove tick labels.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # Remove the 3D panes and grid.
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    # Remove axis lines.
    ax._axis3don = False


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


def plot_cubes(ax, vertices, title="", facecolor="white"):
    """
    Plot each cube face-by-face using its 8 vertices.
    The subplot title is generic (e.g. "Option A") without revealing transformation details.
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
                facecolors=facecolor,
                edgecolors="black",
                alpha=1.0,
            )
            ax.add_collection3d(polygon)
    ax.set_title(title, fontsize=12)
    set_axes_equal(ax, vertices)


def generate_shape_vertices(shape_name, cube_size=1.0):
    """
    Return an array of all corner vertices for the given shape name.
    """
    if shape_name not in SHAPES:
        raise ValueError(f"Unknown shape {shape_name}")
    cube_origins = SHAPES[shape_name]
    all_vertices = []
    for origin in cube_origins:
        corners = cube_vertices(origin, size=cube_size)
        all_vertices.append(corners)
    return np.vstack(all_vertices)


def get_transformed_candidate(transformation_func, original, max_attempts=10):
    """
    Apply the transformation function (here, rotation) to the original vertices until
    the candidate differs from the original (within a tolerance).
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
    Uses rotation angles from a subset of possible angles.
    """
    angle_x = np.deg2rad(np.random.choice([0, 60, 90, 120]))
    angle_y = np.deg2rad(np.random.choice([0, 60, 90, 120]))
    angle_z = np.deg2rad(np.random.choice([0, 60, 90, 120]))
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


def get_visually_similar_candidate(chosen_shape_name, original_vertices, cube_size=1.0):
    """
    Return a candidate from the similar mapping that is visually similar to the chosen shape.
    The candidate is generated from a shape listed in SIMILAR_MAPPING and then rotated.
    If no mapping candidate is available, return None.
    """
    if chosen_shape_name in SIMILAR_MAPPING:
        similar_candidates = SIMILAR_MAPPING[chosen_shape_name][:]
        random.shuffle(similar_candidates)
        for similar_shape_name in similar_candidates:
            if similar_shape_name in SHAPES:
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
                    return candidate_vertices
    return None


def generate_one_image(index, facecolor="white"):
    """
    Generate a single image with a complex polycube shape and four candidate rotations.
    The correct candidate is produced by a pure rotation of the original shape.
    Three wrong candidates are produced:
      - One by mirroring the original shape (then rotated).
      - One by using a similar object (from the mapping, rotated).
      - One by a second mirrored-rotated transformation.
    The candidate order is shuffled and no transformation details are displayed.
    Metadata is saved to metadata.jsonl.
    """
    cube_size = 1.0
    shapes_list = list(SHAPES.keys())
    shape_name = random.choice(shapes_list)
    original_vertices = generate_shape_vertices(shape_name, cube_size=cube_size)

    # Correct candidate: pure rotation.
    correct_candidate = get_transformed_candidate(transform_rotate, original_vertices)

    # Wrong candidate: mirror the original shape.
    mirror_candidate = get_transformed_candidate(
        transform_rotate, transform_mirror(original_vertices)
    )

    # Wrong candidate: similar object from mapping.
    similar_candidate = get_visually_similar_candidate(
        shape_name, original_vertices, cube_size
    )
    if similar_candidate is None:
        similar_candidate = mirror_candidate

    # Additional wrong candidate: another mirrored-rotated version.
    mirror_candidate2 = get_transformed_candidate(
        transform_rotate, transform_mirror(original_vertices)
    )

    candidates = [
        ("rotate", correct_candidate),
        ("mirror", mirror_candidate),
        ("similar", similar_candidate),
        ("mirror2", mirror_candidate2),
    ]
    random.shuffle(candidates)
    correct_candidate_index = [
        i for i, cand in enumerate(candidates) if cand[0] == "rotate"
    ][0]

    # Use a larger figure with 4 candidate options.
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 4, height_ratios=[0.5, 1], wspace=0.1, hspace=0.1)

    ax_orig = fig.add_subplot(gs[0, :], projection="3d")
    plot_cubes(ax_orig, original_vertices, title="Original Shape", facecolor=facecolor)

    for i in range(4):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        _, candidate_vertices = candidates[i]
        plot_cubes(
            ax, candidate_vertices, title=f"Option {chr(65 + i)}", facecolor=facecolor
        )

    filename = f"{index}_{shape_name}.png"
    outdir = OUTDIR
    output_path = os.path.join(outdir, filename)
    plt.savefig(output_path, dpi=60, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    metadata = {
        "filename": filename,
        "candidate_order": [tag for tag, _ in candidates],
        "answer": chr(65 + correct_candidate_index),
    }
    with open(os.path.join(outdir, "metadata.jsonl"), "a") as f:
        f.write(json.dumps(metadata) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple mental-rotation test images of complex polycubes."
    )
    parser.add_argument(
        "--num_images", "-n", type=int, default=1, help="Number of images to generate."
    )
    parser.add_argument(
        "--color", "-c", type=str, default="white", help="Color for the polycubes."
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=69, help="Seed for reproducible results."
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        default="data/mrt/complex",
        help="Output directory for images and metadata.",
    )

    args = parser.parse_args()

    global OUTDIR
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    # Set the seed for reproducibility if provided.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    for i in range(args.num_images):
        generate_one_image(i, facecolor=args.color)


if __name__ == "__main__":
    main()
