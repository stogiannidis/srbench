from PIL import Image, ImageDraw, ImageFont
import random
import argparse
import os
import json
import math
from collections import defaultdict

OUTPUT_DIR = "data/pf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_paper(draw):
    """Draw a 100x100 white square paper with a black outline."""
    draw.rectangle((10, 10, 110, 110), outline="black", fill="white")


def draw_dashed_vertical(draw, x, y1, y2, dash_length=4, gap_length=4):
    y = y1
    while y < y2:
        draw.line([(x, y), (x, min(y + dash_length, y2))], fill="black", width=1)
        y += dash_length + gap_length


def draw_dashed_horizontal(draw, y, x1, x2, dash_length=4, gap_length=4):
    x = x1
    while x < x2:
        draw.line([(x, y), (min(x + dash_length, x2), y)], fill="black", width=1)
        x += dash_length + gap_length


def draw_dashed_diagonal(draw, x1, y1, x2, y2, dash_length=4, gap_length=4):
    total_length = x2 - x1
    pos = 0
    while pos < total_length:
        dash_end = min(pos + dash_length, total_length)
        draw.line(
            [(x1 + pos, y1 + pos), (x1 + dash_end, y1 + dash_end)],
            fill="black",
            width=1,
        )
        pos += dash_length + gap_length


def draw_dashed_negative_diagonal(draw, x1, y1, x2, y2, dash_length=4, gap_length=4):
    total_length = x1 - x2
    pos = 0
    while pos < total_length:
        dash_end = min(pos + dash_length, total_length)
        draw.line(
            [(x1 - pos, y1 + pos), (x1 - dash_end, y1 + dash_end)],
            fill="black",
            width=1,
        )
        pos += dash_length + gap_length


def draw_shading(draw, folds):
    for fold in folds:
        if fold == "V":
            draw.rectangle((10, 10, 60, 110), fill="lightgray", outline="black")
        elif fold == "H":
            draw.rectangle((10, 10, 110, 60), fill="lightgray", outline="black")
        elif fold == "D":
            draw.polygon(
                [(10, 110), (10, 10), (110, 110)], fill="lightgray", outline="black"
            )
        elif fold == "N":
            draw.polygon(
                [(110, 110), (10, 110), (110, 10)], fill="lightgray", outline="black"
            )


def draw_holes(draw, holes):
    """
    Draw holes as small circles. If duplicate coordinates occur,
    offset them slightly in a circular pattern so that all are visible.
    """
    groups = defaultdict(list)
    # Group holes by rounded coordinates.
    for h in holes:
        key = (round(h[0], 1), round(h[1], 1))
        groups[key].append(h)
    for key, group in groups.items():
        n = len(group)
        if n == 1:
            x, y = group[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="black")
        else:
            # Distribute duplicates in a circle of small radius.
            radius_offset = 2
            for i, h in enumerate(group):
                angle = 2 * math.pi * i / n
                offset_x = radius_offset * math.cos(angle)
                offset_y = radius_offset * math.sin(angle)
                x, y = h[0] + offset_x, h[1] + offset_y
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="black")


# Reflection logic as in your original code.
fold_reflections = {
    "V": lambda p: (120 - p[0], p[1]),  # Reflect over x=60
    "H": lambda p: (p[0], 120 - p[1]),  # Reflect over y=60
    "D": lambda p: (p[1], p[0]),  # Reflect over diagonal y=x
    "N": lambda p: (120 - p[1], 120 - p[0]),  # Reflect about y=-x through (60,60)
}


def compute_all_layers(punched_holes, folds):
    """
    Compute unfolded holes by doubling the layers for each fold.
    Each fold produces two layers from every existing layer.
    """
    layers = [punched_holes]
    for fold in folds:
        new_layers = []
        for layer in layers:
            new_layers.append(layer)
            new_layers.append([fold_reflections[fold](h) for h in layer])
        layers = new_layers
    # Flatten layers; note that duplicates are allowed.
    return [h for layer in layers for h in layer]


# Helper function: point in polygon using ray-casting.
def point_in_poly(x, y, poly):
    num = len(poly)
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and (
            x
            < (poly[j][0] - poly[i][0])
            * (y - poly[i][1])
            / (poly[j][1] - poly[i][1] + 1e-10)
            + poly[i][0]
        ):
            c = not c
        j = i
    return c


def generate_hole_in_poly(poly, margin=5):
    """
    Generate a random point (hole) within the polygon 'poly',
    ensuring it lies at least 'margin' away from the polygon's bounding box edges.
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bx_min, bx_max = min(xs) + margin, max(xs) - margin
    by_min, by_max = min(ys) + margin, max(ys) - margin
    while True:
        x = random.uniform(bx_min, bx_max)
        y = random.uniform(by_min, by_max)
        if point_in_poly(x, y, poly):
            return (int(round(x)), int(round(y)))


# Modified clip_polygon: force the fold to occur at the midpoint.
def clip_polygon(poly, fold):
    new_poly = []
    if fold == "V":
        bx_min = min(p[0] for p in poly)
        bx_max = max(p[0] for p in poly)
        mid = (bx_min + bx_max) / 2
        inside = lambda p: p[0] >= mid

        def compute_intersection(p1, p2):
            t = (mid - p1[0]) / (p2[0] - p1[0])
            return (mid, p1[1] + t * (p2[1] - p1[1]))
    elif fold == "H":
        by_min = min(p[1] for p in poly)
        by_max = max(p[1] for p in poly)
        mid = (by_min + by_max) / 2
        inside = lambda p: p[1] >= mid

        def compute_intersection(p1, p2):
            t = (mid - p1[1]) / (p2[1] - p1[1])
            return (p1[0] + t * (p2[0] - p1[0]), mid)
    elif fold == "D":
        bx_min = min(p[0] for p in poly)
        bx_max = max(p[0] for p in poly)
        by_min = min(p[1] for p in poly)
        by_max = max(p[1] for p in poly)
        cx = (bx_min + bx_max) / 2
        cy = (by_min + by_max) / 2
        b = cy - cx  # fold-line: y = x + (cy - cx)
        if sum(1 for p in poly if abs(p[1] - (p[0] + b)) < 1e-3) >= 2:
            return clip_polygon(poly, "N")
        inside = lambda p: p[1] <= p[0] + b

        def compute_intersection(p1, p2):
            denom = (p2[0] - p1[0]) - (p2[1] - p1[1])
            if denom == 0:
                return p1
            t = (p1[1] - p1[0] - b) / denom
            return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    elif fold == "N":
        bx_min = min(p[0] for p in poly)
        bx_max = max(p[0] for p in poly)
        by_min = min(p[1] for p in poly)
        by_max = max(p[1] for p in poly)
        cx = (bx_min + bx_max) / 2
        cy = (by_min + by_max) / 2
        inside = lambda p: (p[0] + p[1]) <= (cx + cy)

        def compute_intersection(p1, p2):
            denom = (p2[0] - p1[0]) + (p2[1] - p1[1])
            if denom == 0:
                return p1
            t = ((cx + cy) - (p1[0] + p1[1])) / denom
            return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    else:
        return poly

    for i in range(len(poly)):
        curr = poly[i]
        prev = poly[i - 1]
        if inside(curr):
            if not inside(prev):
                new_poly.append(compute_intersection(prev, curr))
            new_poly.append(curr)
        elif inside(prev):
            new_poly.append(compute_intersection(prev, curr))
    return new_poly


# Global list to collect final view images for each fold.
process_images = []


def recursive_fold(current_folds, idx, poly):
    """
    Recursively generate the final folded view for each fold.
    'poly' represents the current visible paper shape.
    Only the final view after each fold is saved.
    Returns the final polygon after all folds.
    """
    if idx == 0:
        img_unfolded = Image.new("RGB", (120, 120), "white")
        draw_unfolded = ImageDraw.Draw(img_unfolded)
        draw_paper(draw_unfolded)
        process_images.append((img_unfolded, "Unfolded"))
    if idx >= len(current_folds):
        return poly
    fold = current_folds[idx]
    new_poly = clip_polygon(poly, fold)
    img_result = Image.new("RGB", (120, 120), "white")
    draw_result = ImageDraw.Draw(img_result)
    draw_result.polygon(poly, fill="lightgray", outline="black")
    if new_poly:
        draw_result.polygon(new_poly, fill="white", outline="black")
    process_images.append((img_result, f"Fold {idx + 1}"))
    return recursive_fold(current_folds, idx + 1, new_poly)


def generate_wrong_option_remove_one(holes):
    import copy

    candidate = copy.copy(holes)
    if len(candidate) > 1:
        idx = random.randrange(len(candidate))
        candidate.pop(idx)
    elif len(candidate) == 1:
        x, y = candidate[0]
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        candidate[0] = (x + dx, y + dy)
    return candidate


def generate_wrong_option_mirror(holes):
    candidate = []
    for x, y in holes:
        new_x = 120 - x
        new_y = 120 - y
        candidate.append((new_x, new_y))
    return candidate


def generate_wrong_option_rotate(holes):
    candidate = []
    for x, y in holes:
        tx, ty = x - 60, y - 60
        rx, ry = ty, -tx
        candidate.append((rx + 60, ry + 60))
    return candidate


def generate_test_image(folds, test_number, num_folds, num_holes):
    """
    Generate a test image with:
      - Top row: final views after each fold.
      - Bottom row: three candidate options.
    Holes are generated inside the final folded view (with a margin) and then unfolded
    using the layered reflection (doubling layers per fold).
    """
    small_size = 120
    global process_images
    process_images = []  # Reset for each test image

    try:
        # Use a font known to be available on most systems.
        font_bigger = ImageFont.truetype("Arial", size=18)
    except OSError:
        font_bigger = ImageFont.load_default()

    initial_poly = [(10, 10), (110, 10), (110, 110), (10, 110)]
    final_poly = recursive_fold(folds, 0, initial_poly)

    # Generate punched holes within the final polygon.
    punched_holes = [
        generate_hole_in_poly(final_poly, margin=5) for _ in range(num_holes)
    ]
    # Compute unfolded holes (all layers) for the candidate.
    unfolded_holes = compute_all_layers(punched_holes, folds)

    img_final = Image.new("RGB", (small_size, small_size), "white")
    draw_final = ImageDraw.Draw(img_final)
    if final_poly:
        draw_final.polygon(final_poly, fill="white", outline="black")
    draw_holes(draw_final, punched_holes)
    process_images.append((img_final, "Final view"))

    # Correct (unfolded) option.
    img_correct = Image.new("RGB", (small_size, small_size), "white")
    draw_correct = ImageDraw.Draw(img_correct)
    draw_paper(draw_correct)
    draw_holes(draw_correct, unfolded_holes)

    wrong_b = generate_wrong_option_remove_one(unfolded_holes)
    img_wrong_b = Image.new("RGB", (small_size, small_size), "white")
    draw_wrong_b = ImageDraw.Draw(img_wrong_b)
    draw_paper(draw_wrong_b)
    draw_holes(draw_wrong_b, wrong_b)

    wrong_c = generate_wrong_option_mirror(unfolded_holes)
    if set(wrong_c) == set(unfolded_holes):
        wrong_c = generate_wrong_option_rotate(unfolded_holes)
        if set(wrong_c) == set(unfolded_holes):
            wrong_c = [(min(x + 5, 110), y) for (x, y) in unfolded_holes]
    img_wrong_c = Image.new("RGB", (small_size, small_size), "white")
    draw_wrong_c = ImageDraw.Draw(img_wrong_c)
    draw_paper(draw_wrong_c)
    draw_holes(draw_wrong_c, wrong_c)

    candidates = [
        ("correct", img_correct),
        ("wrong", img_wrong_b),
        ("wrong", img_wrong_c),
    ]
    random.shuffle(candidates)
    option_labels = ["A", "B", "C"]
    options = []
    correct_label = None
    for label, (kind, img) in zip(option_labels, candidates):
        options.append((img, label))
        if kind == "correct":
            correct_label = label

    top_width = len(process_images) * small_size + (len(process_images) - 1) * 10
    bottom_width = 3 * small_size + 2 * 10
    total_width = max(top_width, bottom_width)
    total_height = 320  # Set overall canvas height to 320
    total_img = Image.new("RGB", (total_width, total_height), "white")
    draw_total = ImageDraw.Draw(total_img)

    start_x = (total_width - top_width) // 2
    current_x = start_x
    # Draw top row images and text; text drawn at y=10.
    for img, label in process_images:
        total_img.paste(img, (current_x, 30))
        draw_total.text(
            (current_x + 15, 10), label, fill="black", font=font_bigger, align="center"
        )
        current_x += small_size + 10

    start_x_bottom = (total_width - bottom_width) // 2
    current_x = start_x_bottom
    # Draw bottom row images and text; images now pasted at y=170 and text at y=190.
    for img, label in options:
        total_img.paste(img, (current_x, 190))
        draw_total.text(
            (current_x + 45, 180), label, fill="black", font=font_bigger, align="center"
        )
        current_x += small_size + 10

    f_name = f"{test_number}_fold-{num_folds}_holes-{num_holes}.png"
    out_path = os.path.join(OUTPUT_DIR, f_name)
    total_img.save(out_path)
    return out_path, correct_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate folded paper images with holes."
    )
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=10,
        help="Number of images to generate (default: 10)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator (default: 42)",
    )
    parser.add_argument(
        "-f",
        "--num-folds",
        type=int,
        default=2,
        choices=range(1, 10),
        help="Number of folds (minimum 1, maximum 9) (default: 2)",
    )
    parser.add_argument(
        "-H", "--num-holes", type=int, default=1, help="Number of holes (default: 1)"
    )
    parser.add_argument(
        "-m",
        "--metadata-file",
        type=str,
        default="metadata.jsonl",
        help="Metadata JSONL file to append to (default: metadata.jsonl)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    metadata_path = os.path.join(OUTPUT_DIR, args.metadata_file)
    with open(metadata_path, "a", encoding="utf-8") as metaf:
        for i in range(1, args.num_images + 1):
            fold_group = random.choice(["VH", "Diagonal"])
            if fold_group == "VH":
                folds = [random.choice(["V", "H"]) for _ in range(args.num_folds)]
            else:
                folds = [random.choice(["D", "N"]) for _ in range(args.num_folds)]
            image_path, correct_option = generate_test_image(
                folds, i, args.num_folds, args.num_holes
            )
            rel_image_path = os.path.basename(image_path)
            metadata_obj = {
                "filename": rel_image_path,
                "correct_option": correct_option,
            }
            metaf.write(json.dumps(metadata_obj) + "\n")
