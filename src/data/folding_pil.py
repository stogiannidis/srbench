from PIL import Image, ImageDraw, ImageFont
import random
import argparse
import os
import json
import math
from collections import defaultdict

OUTPUT_DIR = "data/pf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache font loading
_font_cache = {}


def get_font(size=18):
    """Get cached font instance."""
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype("Arial", size=size)
        except OSError:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_paper(draw):
    """Draw a 100x100 white square paper with a black outline."""
    draw.rectangle((10, 10, 110, 110), outline="black", fill="white")


def draw_holes(draw, holes):
    """
    Draw holes as small circles. If duplicate coordinates occur,
    offset them slightly in a circular pattern so that all are visible.
    """
    if not holes:
        return

    groups = defaultdict(list)
    # Group holes by rounded coordinates.
    for h in holes:
        key = (round(h[0], 1), round(h[1], 1))
        groups[key].append(h)

    for group in groups.values():
        n = len(group)
        if n == 1:
            x, y = group[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="black")
        else:
            # Distribute duplicates in a circle of small radius.
            radius_offset = 2
            angle_step = 2 * math.pi / n
            for i, h in enumerate(group):
                angle = angle_step * i
                offset_x = radius_offset * math.cos(angle)
                offset_y = radius_offset * math.sin(angle)
                x, y = h[0] + offset_x, h[1] + offset_y
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="black")


# Reflection logic - use lambda functions for better performance
FOLD_REFLECTIONS = {
    "V": lambda p: (120 - p[0], p[1]),
    "H": lambda p: (p[0], 120 - p[1]),
    "D": lambda p: (p[1], p[0]),
    "N": lambda p: (120 - p[1], 120 - p[0]),
}


def compute_all_layers(punched_holes, folds):
    """
    Compute unfolded holes by doubling the layers for each fold.
    """
    if not punched_holes:
        return []

    layers = [punched_holes]
    for fold in folds:
        reflection_func = FOLD_REFLECTIONS[fold]
        new_layers = []
        for layer in layers:
            new_layers.append(layer)
            new_layers.append([reflection_func(h) for h in layer])
        layers = new_layers

    # Flatten layers
    return [h for layer in layers for h in layer]


def point_in_poly(x, y, poly):
    """Optimized point-in-polygon test using ray casting."""
    if not poly:
        return False

    inside = False
    j = len(poly) - 1

    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def generate_hole_in_poly(poly, margin=5):
    """Generate a random point within the polygon with margin."""
    if not poly:
        return (60, 60)  # Default center point

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bx_min, bx_max = min(xs) + margin, max(xs) - margin
    by_min, by_max = min(ys) + margin, max(ys) - margin

    # Ensure valid bounds
    if bx_min >= bx_max or by_min >= by_max:
        return (int((min(xs) + max(xs)) / 2), int((min(ys) + max(ys)) / 2))

    for _ in range(100):  # Limit attempts to avoid infinite loop
        x = random.uniform(bx_min, bx_max)
        y = random.uniform(by_min, by_max)
        if point_in_poly(x, y, poly):
            return (int(round(x)), int(round(y)))

    # Fallback to polygon centroid
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return (int(round(cx)), int(round(cy)))


def clip_polygon(poly, fold):
    """Optimized polygon clipping."""
    if not poly:
        return []

    # Pre-compute bounding box
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bx_min, bx_max = min(xs), max(xs)
    by_min, by_max = min(ys), max(ys)

    if fold == "V":
        mid = (bx_min + bx_max) / 2
        inside = lambda p: p[0] >= mid
        intersect = lambda p1, p2: (
            mid,
            p1[1] + (mid - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0]),
        )
    elif fold == "H":
        mid = (by_min + by_max) / 2
        inside = lambda p: p[1] >= mid
        intersect = lambda p1, p2: (
            p1[0] + (mid - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]),
            mid,
        )
    elif fold == "D":
        cx, cy = (bx_min + bx_max) / 2, (by_min + by_max) / 2
        b = cy - cx
        inside = lambda p: p[1] <= p[0] + b

        def intersect(p1, p2):
            denom = (p2[0] - p1[0]) - (p2[1] - p1[1])
            if abs(denom) < 1e-10:
                return p1
            t = (p1[1] - p1[0] - b) / denom
            return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    elif fold == "N":
        cx, cy = (bx_min + bx_max) / 2, (by_min + by_max) / 2
        inside = lambda p: (p[0] + p[1]) <= (cx + cy)

        def intersect(p1, p2):
            denom = (p2[0] - p1[0]) + (p2[1] - p1[1])
            if abs(denom) < 1e-10:
                return p1
            t = ((cx + cy) - (p1[0] + p1[1])) / denom
            return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    else:
        return poly

    new_poly = []
    for i in range(len(poly)):
        curr = poly[i]
        prev = poly[i - 1]

        if inside(curr):
            if not inside(prev):
                new_poly.append(intersect(prev, curr))
            new_poly.append(curr)
        elif inside(prev):
            new_poly.append(intersect(prev, curr))

    return new_poly


# Global list to collect final view images for each fold.
process_images = []


def recursive_fold(current_folds, idx, poly):
    """Recursively generate the final folded view for each fold."""
    global process_images

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

    if poly:
        draw_result.polygon(poly, fill="lightgray", outline="black")
    if new_poly:
        draw_result.polygon(new_poly, fill="white", outline="black")

    process_images.append((img_result, f"Fold {idx + 1}"))
    return recursive_fold(current_folds, idx + 1, new_poly)


def generate_wrong_options(holes):
    """Generate all wrong options at once for efficiency."""
    if not holes:
        return [[], []]

    # Option 1: Remove one hole or shift if only one hole
    option1 = holes.copy()
    if len(option1) > 1:
        option1.pop(random.randrange(len(option1)))
    else:
        x, y = option1[0]
        dx, dy = random.randint(-3, 3), random.randint(-3, 3)
        option1[0] = (x + dx, y + dy)

    # Option 2: Mirror transformation
    option2 = [(120 - x, 120 - y) for x, y in holes]

    # If mirror is same as original, try rotation
    if set(option2) == set(holes):
        option2 = [
            (ty + 60, -tx + 60) for x, y in holes for tx, ty in [(x - 60, y - 60)]
        ]
        # If rotation is also same, shift horizontally
        if set(option2) == set(holes):
            option2 = [(min(x + 5, 110), y) for x, y in holes]

    return [option1, option2]


def generate_test_image(folds, test_number, num_folds, num_holes):
    """Generate a test image with optimized processing."""
    global process_images
    process_images = []

    font_bigger = get_font(18)

    initial_poly = [(10, 10), (110, 10), (110, 110), (10, 110)]
    final_poly = recursive_fold(folds, 0, initial_poly)

    # Generate punched holes
    punched_holes = [
        generate_hole_in_poly(final_poly, margin=5) for _ in range(num_holes)
    ]
    unfolded_holes = compute_all_layers(punched_holes, folds)

    # Create final view image
    img_final = Image.new("RGB", (120, 120), "white")
    draw_final = ImageDraw.Draw(img_final)
    if final_poly:
        draw_final.polygon(final_poly, fill="white", outline="black")
    draw_holes(draw_final, punched_holes)
    process_images.append((img_final, "Final view"))

    # Create candidate images
    img_correct = Image.new("RGB", (120, 120), "white")
    draw_correct = ImageDraw.Draw(img_correct)
    draw_paper(draw_correct)
    draw_holes(draw_correct, unfolded_holes)

    wrong_options = generate_wrong_options(unfolded_holes)
    candidate_images = [("correct", img_correct)]

    for wrong_holes in wrong_options:
        img_wrong = Image.new("RGB", (120, 120), "white")
        draw_wrong = ImageDraw.Draw(img_wrong)
        draw_paper(draw_wrong)
        draw_holes(draw_wrong, wrong_holes)
        candidate_images.append(("wrong", img_wrong))

    # Shuffle and assign labels
    random.shuffle(candidate_images)
    option_labels = ["A", "B", "C"]
    correct_label = None

    for i, (kind, img) in enumerate(candidate_images):
        if kind == "correct":
            correct_label = option_labels[i]

    # Create final composite image
    small_size = 120
    top_width = len(process_images) * small_size + (len(process_images) - 1) * 10
    bottom_width = 3 * small_size + 2 * 10
    total_width = max(top_width, bottom_width)
    total_height = 320

    total_img = Image.new("RGB", (total_width, total_height), "white")
    draw_total = ImageDraw.Draw(total_img)

    # Draw top row
    start_x = (total_width - top_width) // 2
    for i, (img, label) in enumerate(process_images):
        x = start_x + i * (small_size + 10)
        total_img.paste(img, (x, 30))
        draw_total.text((x + 15, 10), label, fill="black", font=font_bigger)

    # Draw bottom row
    start_x_bottom = (total_width - bottom_width) // 2
    for i, (_, img) in enumerate(candidate_images):
        x = start_x_bottom + i * (small_size + 10)
        total_img.paste(img, (x, 190))
        draw_total.text((x + 45, 180), option_labels[i], fill="black", font=font_bigger)

    # Save image
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
