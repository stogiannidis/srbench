#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from shapely.ops import split
import math
import random

def plot_line_difference(line, cut_line, style, ax, **kwargs):
    """
    Given a LineString (or MultiLineString) 'line', subtracts the parts that
    intersect with 'cut_line' and plots the remainder with the given style.
    """
    diff_line = line.difference(cut_line)
    if diff_line.is_empty:
        return
    if diff_line.geom_type == 'LineString':
        x, y = diff_line.xy
        ax.plot(x, y, style, **kwargs)
    elif diff_line.geom_type == 'MultiLineString':
        for seg in diff_line.geoms:
            x, y = seg.xy
            ax.plot(x, y, style, **kwargs)

def plot_square_with_dashed_fold(ax, holes, fold_lines=None, folded_region=None, draw_unfolded=True):
    """
    Draws the paper (a unit square). When a folded_region is provided:
      - If draw_unfolded is True, the full paper is drawn with the unfolded part
        shown in a solid black line (except along the crease) and the crease drawn
        in a red dashed line.
      - If draw_unfolded is False, only the folded region (after the fold) is drawn.
    Holes (punctures) and crease lines (if provided) are always drawn.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    if folded_region and not draw_unfolded:
        # Only draw the folded (post-fold) portion.
        folded_poly = Polygon(folded_region)
        x, y = folded_poly.exterior.xy
        ax.plot(x, y, 'r--', linewidth=1)
    elif folded_region:
        # Draw the full paper then subtract the folded region's edge from the unfolded part.
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        folded_poly = Polygon(folded_region)
        unfolded_poly = square.difference(folded_poly)
        folded_boundary = folded_poly.exterior
        if unfolded_poly.geom_type == 'Polygon':
            plot_line_difference(unfolded_poly.exterior, folded_boundary, 'k-', ax, linewidth=1, zorder=1)
        elif unfolded_poly.geom_type == 'MultiPolygon':
            for poly in unfolded_poly:
                plot_line_difference(poly.exterior, folded_boundary, 'k-', ax, linewidth=1, zorder=1)
        x, y = folded_boundary.xy
        ax.plot(x, y, 'r--', linewidth=1, zorder=2)
    else:
        # Draw the full square.
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        x, y = square.exterior.xy
        ax.plot(x, y, 'k-', linewidth=1)

    # Draw crease lines if provided.
    if fold_lines:
        for line in fold_lines:
            ax.plot(line[0], line[1], 'k--', linewidth=1)
    
    # Draw the holes.
    for hole in holes:
        ax.plot(hole[0], hole[1], 'o', markersize=5, color='black')

def rotate_points(points, angle, center=(0.5, 0.5)):
    """
    Rotates a list of (x, y) points about a given center by 'angle' (in degrees).
    """
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = center
    rotated = []
    for (x, y) in points:
        dx = x - cx
        dy = y - cy
        x_new = cx + dx * cos_a - dy * sin_a
        y_new = cy + dx * sin_a + dy * cos_a
        rotated.append((x_new, y_new))
    return rotated

def generate_candidates(correct_holes):
    """
    Generates 4 candidate hole configurations by applying random rotations to the
    correct unfolded holes (a list of points). One candidate is exactly correct (0° rotation)
    and is inserted at a random index.
    
    Returns a tuple (candidates, correct_index).
    """
    candidates = []
    correct_index = random.randint(0, 3)
    for i in range(4):
        if i == correct_index:
            angle = 0
        else:
            angle = random.uniform(-20, 20)
            while abs(angle) < 5:
                angle = random.uniform(-20, 20)
        candidate = rotate_points(correct_holes, angle, center=(0.5, 0.5))
        candidates.append(candidate)
    return candidates, correct_index

def random_point_on_side(side):
    """
    Returns a random point on the specified side of the unit square.
    """
    if side == "top":
        return (random.uniform(0, 1), 1)
    elif side == "bottom":
        return (random.uniform(0, 1), 0)
    elif side == "left":
        return (0, random.uniform(0, 1))
    elif side == "right":
        return (1, random.uniform(0, 1))

def random_point_in_polygon(poly):
    """
    Returns a random point inside the given shapely polygon using rejection sampling.
    """
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return (p.x, p.y)

def reflect_point(P, A, B):
    """
    Reflects point P (tuple) across the line defined by points A and B.
    Uses the standard formula for reflection across a line.
    """
    px, py = P
    ax, ay = A
    bx, by = B
    # Compute vector AB.
    ABx = bx - ax
    ABy = by - ay
    # Compute vector AP.
    APx = px - ax
    APy = py - ay
    # Projection factor.
    t = (APx * ABx + APy * ABy) / (ABx**2 + ABy**2)
    # Projection point.
    proj_x = ax + t * ABx
    proj_y = ay + t * ABy
    # Reflection: R = 2 * proj - P.
    rx = 2 * proj_x - px
    ry = 2 * proj_y - py
    return (rx, ry)

def sample_valid_puncture(folded_poly, crease_point1, crease_point2, square):
    """
    Samples a puncture from the folded region such that both the puncture and its
    reflection across the crease (defined by crease_point1 and crease_point2) are inside
    the unit square.
    """
    while True:
        hole = random_point_in_polygon(folded_poly)
        reflected = reflect_point(hole, crease_point1, crease_point2)
        if square.contains(Point(reflected)):
            return hole, reflected

def create_one_image(image_index, num_punctures):
    """
    Creates one image with:
      - Top row: The original paper and the folded & punched paper.
      - Bottom row: 4 candidate unfolded answers (rotated versions of the correct answer).
    
    The crease is generated by choosing two random sides and picking a random point on each.
    The paper is then split along the crease and one of the resulting regions is randomly chosen
    as the folded region. A total of 'num_punctures' punctures are placed in that region and each
    is reflected across the crease (using repeated sampling if necessary) so that both holes lie
    within the unit square.
    """
    # Create the full paper (unit square).
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    
    # Randomly choose two distinct sides.
    sides = ["top", "bottom", "left", "right"]
    side1 = random.choice(sides)
    side2 = random.choice([s for s in sides if s != side1])
    point1 = random_point_on_side(side1)
    point2 = random_point_on_side(side2)
    
    # The crease line is defined by these two points.
    crease_line = LineString([point1, point2])
    fold_line = ([point1[0], point2[0]], [point1[1], point2[1]])
    
    # Split the square by the crease.
    splitted = list(split(square, crease_line).geoms)
    if len(splitted) < 2:
        # Fallback: if splitting fails, use a vertical fold.
        folded_poly = Polygon([(0.5, 0), (1, 0), (1, 1), (0.5, 1)])
    else:
        folded_poly = random.choice(splitted)
    folded_region = list(folded_poly.exterior.coords)
    
    # Generate the desired number of punctures, ensuring both the original and reflected
    # punctures are within the unit square.
    folded_holes = []
    reflected_holes = []
    for _ in range(num_punctures):
        hole, reflected = sample_valid_puncture(folded_poly, point1, point2, square)
        folded_holes.append(hole)
        reflected_holes.append(reflected)
    
    # The correct unfolded configuration includes all punctures and their reflections.
    correct_unfolded_holes = folded_holes + reflected_holes
    
    # Create a 2x4 grid for the image.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # --- Top Row: Setup Images ---
    # Left: Original paper.
    plot_square_with_dashed_fold(axes[0, 0], [])
    axes[0, 0].set_title("Original Paper")
    
    # Middle: Folded & Punched paper.
    # Draw only the folded region (with red dashed boundary) plus the crease and the folded punctures.
    plot_square_with_dashed_fold(
        axes[0, 1],
        folded_holes,  # Only the punctures in the folded region.
        fold_lines=[fold_line],
        folded_region=folded_region,
        draw_unfolded=False
    )
    axes[0, 1].set_title("Folded & Punched")
    
    # Turn off the remaining top row axes.
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    
    # --- Bottom Row: Candidate Unfolded Answers ---
    candidates, correct_index = generate_candidates(correct_unfolded_holes)
    candidate_titles = ["Option A", "Option B", "Option C", "Option D"]
    for i in range(4):
        ax = axes[1, i]
        # Draw the full unfolded paper with all punctures.
        plot_square_with_dashed_fold(ax, candidates[i])
        title_text = candidate_titles[i]
        if i == correct_index:
            title_text += " (Correct)"
        ax.set_title(title_text)
    
    plt.tight_layout()
    filename = f"generated_image_{image_index+1}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close(fig)

def main(num_images, num_punctures):
    """
    Generate 'num_images' images, each containing 'num_punctures' punctures.
    """
    for i in range(num_images):
        create_one_image(i, num_punctures)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate folding images with random crease, punctures, and candidate unfolded answers."
    )
    parser.add_argument(
        "-k", "--num_images",
        type=int,
        default=5,
        help="Number of images to generate (default: 5)"
    )
    parser.add_argument(
        "-p", "--num_punctures",
        type=int,
        default=1,
        help="Number of punctures per image (default: 1)"
    )
    args = parser.parse_args()
    main(args.num_images, args.num_punctures)
