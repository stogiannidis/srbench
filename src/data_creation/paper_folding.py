import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import argparse
import random
import os
import itertools
import math
import json

# --- Helper functions for polygon clipping, area, and sampling ---


def clip_polygon(poly, f):
    """
    Clips a convex polygon (list of (x,y) vertices in order) with the half‐plane f(p) <= 0.
    Returns the new polygon (as a list of (x,y)).
    f is a function that computes a signed value.
    """
    if not poly:
        return []
    new_poly = []
    n = len(poly)
    for i in range(n):
        p_curr = poly[i]
        p_next = poly[(i + 1) % n]
        f_curr = f(p_curr)
        f_next = f(p_next)
        if f_curr <= 0:
            new_poly.append(p_curr)
        if f_curr * f_next < 0:
            t = f_curr / (f_curr - f_next)
            inter_x = p_curr[0] + t * (p_next[0] - p_curr[0])
            inter_y = p_curr[1] + t * (p_next[1] - p_curr[1])
            new_poly.append((inter_x, inter_y))
    return new_poly


def polygon_area(poly):
    """Compute the area of a polygon using the shoelace formula."""
    area = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2.0


def polygon_bounding_box(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def sample_point_in_polygon(poly):
    """
    Samples a random point uniformly in a convex polygon.
    The polygon is triangulated by taking the first vertex and all consecutive pairs.
    Then a triangle is chosen with probability proportional to its area.
    """
    if len(poly) < 3:
        return poly[0]
    triangles = []
    areas = []
    A = poly[0]
    for i in range(1, len(poly) - 1):
        B = poly[i]
        C = poly[i + 1]
        triangles.append((A, B, C))
        area = abs((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])) / 2.0
        areas.append(area)
    total_area = sum(areas)
    r = random.uniform(0, total_area)
    cum = 0
    chosen = triangles[0]
    for tri, area in zip(triangles, areas):
        cum += area
        if r <= cum:
            chosen = tri
            break
    u = random.random()
    v = random.random()
    if u + v > 1:
        u, v = 1 - u, 1 - v
    A, B, C = chosen
    x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
    y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
    return (x, y)


def rotate_point(p, angle_deg, center=(0.5, 0.5)):
    """
    Rotate point p by angle_deg degrees around the given center.
    """
    angle_rad = math.radians(angle_deg)
    x, y = p
    cx, cy = center
    x -= cx
    y -= cy
    x_new = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_new = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return (x_new + cx, y_new + cy)


def clamp_point(p, min_val=0.0, max_val=1.0):
    """
    Clamp point p so that both coordinates are between min_val and max_val.
    """
    x, y = p
    return (max(min_val, min(x, max_val)), max(min_val, min(y, max_val)))


# --- Functions for generating folds on a polygon with reasonable area ---


def generate_folds_poly(num_folds, area_threshold=0.3, max_attempts=10):
    """
    Starting with the unit square as a polygon, apply num_folds sequentially.
    Each fold is randomly chosen among 'vertical', 'horizontal', and 'diagonal'.
    For vertical/horizontal folds, we use the midpoint of the current bounding box.
    For diagonal folds, we randomly choose one of the two diagonals.
    To ensure a "reasonable" fold, we require that the area of the resulting polygon
    is at least area_threshold (e.g. 30%) of the original polygon's area.
    If a fold attempt reduces the area too much, try another fold type (up to max_attempts).
    Returns a list of fold dictionaries (with fold parameters) and a list of intermediate polygon states.
    """
    poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
    states = [poly]
    folds = []
    for i in range(num_folds):
        old_area = polygon_area(poly)
        best_poly = None
        best_area = -1
        best_fold = None
        for attempt in range(max_attempts):
            fold_type = random.choice(["vertical", "horizontal", "diagonal"])
            bbox = polygon_bounding_box(poly)
            min_x, min_y, max_x, max_y = bbox
            if fold_type == "vertical":
                crease = (min_x + max_x) / 2.0
                f_func = lambda p: p[0] - crease
                new_poly = clip_polygon(poly, f_func)
                fold_info = {"type": "vertical", "crease": crease}
            elif fold_type == "horizontal":
                crease = (min_y + max_y) / 2.0
                f_func = lambda p: p[1] - crease
                new_poly = clip_polygon(poly, f_func)
                fold_info = {"type": "horizontal", "crease": crease}
            else:
                direction = random.choice(["d1", "d2"])
                if direction == "d1":
                    A = (min_x, min_y)
                    B = (max_x, max_y)

                    def f_func(p, A=A, B=B):
                        return (B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (
                            p[0] - A[0]
                        )

                    new_poly = clip_polygon(poly, f_func)
                    fold_info = {"type": "diagonal", "direction": "d1", "A": A, "B": B}
                else:
                    A = (min_x, max_y)
                    B = (max_x, min_y)

                    def f_func(p, A=A, B=B):
                        return (B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (
                            p[0] - A[0]
                        )

                    new_poly = clip_polygon(poly, f_func)
                    fold_info = {"type": "diagonal", "direction": "d2", "A": A, "B": B}
            new_area = polygon_area(new_poly)
            if new_area >= area_threshold * old_area:
                best_poly = new_poly
                best_area = new_area
                best_fold = fold_info
                break
            elif new_area > best_area:
                best_poly = new_poly
                best_area = new_area
                best_fold = fold_info
        poly = best_poly
        folds.append(best_fold)
        states.append(poly)
    return folds, states


# --- Unfolding function (supports vertical, horizontal, and diagonal) ---


def unfold_point_general(p, folds, decisions):
    """
    Given a point p in the final folded region and a decision vector (one boolean per fold),
    compute the unfolded coordinate by processing folds in reverse order.
    For each fold with decision True, reflect p across that fold’s crease.
    Supports 'vertical', 'horizontal', and 'diagonal' folds.
    """
    x, y = p
    for fold, decision in reversed(list(zip(folds, decisions))):
        if not decision:
            continue
        if fold["type"] == "vertical":
            crease = fold["crease"]
            x = crease + (crease - x)
        elif fold["type"] == "horizontal":
            crease = fold["crease"]
            y = crease + (crease - y)
        elif fold["type"] == "diagonal":
            A = fold["A"]
            B = fold["B"]
            d = (B[0] - A[0], B[1] - A[1])
            u = (x - A[0], y - A[1])
            dot = u[0] * d[0] + u[1] * d[1]
            d_dot = d[0] * d[0] + d[1] * d[1]
            if d_dot == 0:
                continue
            proj = (dot / d_dot * d[0], dot / d_dot * d[1])
            x = A[0] + 2 * proj[0] - u[0]
            y = A[1] + 2 * proj[1] - u[1]
    return (x, y)


# --- Main function for generating images and metadata ---


def main(num_folds, num_puncture, num_img):
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    metadata_filename = os.path.join(output_dir, "metadata.jsonl")
    metafile = open(metadata_filename, "w")

    for img_index in range(num_img):
        # Generate fold sequence and intermediate polygon states.
        folds, states = generate_folds_poly(num_folds)
        final_poly = states[-1]

        # Generate random punctures uniformly in the final folded region.
        punctures_folded = [
            sample_point_in_polygon(final_poly) for _ in range(num_puncture)
        ]

        # Compute all possible unfolded positions.
        if num_folds > 0:
            all_decisions = list(itertools.product([True, False], repeat=num_folds))
        else:
            all_decisions = [()]
        unfolded_all = set()
        for p in punctures_folded:
            for decisions in all_decisions:
                up = unfold_point_general(p, folds, decisions)
                unfolded_all.add(up)
        unfolded_all = list(unfolded_all)

        # Build candidate sets.
        candidate_correct = unfolded_all
        candidate_offset1 = [
            clamp_point((x + 0.02, y + 0.02)) for (x, y) in unfolded_all
        ]
        candidate_offset2 = [
            clamp_point((x - 0.02, y - 0.02)) for (x, y) in unfolded_all
        ]
        candidate_rotated = [
            clamp_point(rotate_point(p, 5, center=(0.5, 0.5))) for p in unfolded_all
        ]

        candidates = [
            (candidate_correct, "correct"),
            (candidate_offset1, "offset1"),
            (candidate_offset2, "offset2"),
            (candidate_rotated, "rotated"),
        ]
        random.shuffle(candidates)
        for idx, (cand, label) in enumerate(candidates):
            if label == "correct":
                correct_candidate_index = idx + 1  # 1-indexed
                break

        # --- Create the figure ---
        # Layout:
        # TOP ROW: Horizontal row of folding process (each intermediate state side-by-side),
        #          with the final state showing the puncture(s).
        # BOTTOM ROW: Four candidate unfolded views with overlaid text labels.
        fig = plt.figure(figsize=(18, 10))
        outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # TOP ROW: Arrange intermediate states horizontally.
        num_states = len(states)
        process_gs = gridspec.GridSpecFromSubplotSpec(
            1, num_states, subplot_spec=outer[0]
        )
        for i, state in enumerate(states):
            ax = plt.Subplot(fig, process_gs[i])
            ax.add_patch(
                patches.Polygon(
                    state, closed=True, fill=False, edgecolor="black", linewidth=2
                )
            )
            # For i > 0, draw the crease line from the fold that produced this state in red.
            if i > 0:
                fold = folds[i - 1]
                if fold["type"] == "vertical":
                    bbox = polygon_bounding_box(state)
                    min_y, max_y = bbox[1], bbox[3]
                    crease = fold["crease"]
                    ax.plot(
                        [crease, crease], [min_y, max_y], linestyle="--", color="red"
                    )
                elif fold["type"] == "horizontal":
                    bbox = polygon_bounding_box(state)
                    min_x, max_x = bbox[0], bbox[2]
                    crease = fold["crease"]
                    ax.plot(
                        [min_x, max_x], [crease, crease], linestyle="--", color="red"
                    )
                elif fold["type"] == "diagonal":
                    A = fold["A"]
                    B = fold["B"]
                    ax.plot([A[0], B[0]], [A[1], B[1]], linestyle="--", color="red")
            # If this is the final state, plot the punctures.
            if i == num_states - 1:
                for p in punctures_folded:
                    ax.plot(p[0], p[1], "ko", markersize=6)
            ax.set_title(f"Fold {i}", fontsize=18, color="black")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.add_subplot(ax)

        # BOTTOM ROW: Candidate unfolded views with text labels.
        bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1])
        for i, (cand, _) in enumerate(candidates):
            ax = plt.Subplot(fig, bottom_gs[i])
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black"))
            for up in cand:
                ax.plot(up[0], up[1], "ko", markersize=6)
            # Overlay candidate label text on the image.
            ax.text(
                0.5,
                0.95,
                f"Candidate {i + 1}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=18,
                color="black",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.add_subplot(ax)

        plt.tight_layout()
        filename = os.path.join(output_dir, f"paper_folding_image_{img_index + 1}.png")
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved image: {filename}")

        # Write metadata.
        metadata = {
            "filename": filename,
            "candidate_order": [label for (_, label) in candidates],
            "correct_candidate": correct_candidate_index,  # 1-indexed
        }
        metafile.write(json.dumps(metadata) + "\n")

    metafile.close()
    print(f"Metadata saved to {metadata_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paper folding simulation images with vertical, horizontal, and diagonal folds. "
        "The folding process is showcased horizontally and punctures appear in the final state. "
        "Four candidate unfolded views (with labels) are produced; candidate points are clamped to the paper boundaries. "
        "Dashed folding lines are drawn in red, all text is fontsize 18 in black, and all axes are set to 0–1."
    )
    parser.add_argument(
        "--num_folds", type=int, default=2, help="Number of folds to apply."
    )
    parser.add_argument(
        "--num_puncture", type=int, default=1, help="Number of punctures to draw."
    )
    parser.add_argument(
        "--num_img", type=int, default=1, help="Number of images to generate."
    )
    args = parser.parse_args()

    main(args.num_folds, args.num_puncture, args.num_img)
