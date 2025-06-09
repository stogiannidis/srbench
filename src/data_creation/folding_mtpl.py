#!/usr/bin/env python3
"""
Paper Folding Test Generator using matplotlib.

Generates paper folding visualization images with holes punched through folded paper.
Uses matplotlib for all drawing operations instead of PIL/Pillow.
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float
    
    def __iter__(self):
        yield self.x
        yield self.y


@dataclass
class Config:
    """Configuration for paper folding generation."""
    paper_size: float = 100.0
    paper_margin: float = 10.0
    hole_radius: float = 2.0
    hole_margin: float = 5.0
    dash_length: float = 4.0
    gap_length: float = 4.0
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    linewidth: float = 2.0


class GeometryUtils:
    """Utility functions for geometric operations."""
    
    @staticmethod
    def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def get_polygon_bounds(polygon: List[Point]) -> Tuple[float, float, float, float]:
        """Get bounding box of polygon: (min_x, max_x, min_y, max_y)."""
        if not polygon:
            return 0, 0, 0, 0
        
        xs = [p.x for p in polygon]
        ys = [p.y for p in polygon]
        return min(xs), max(xs), min(ys), max(ys)
    
    @staticmethod
    def generate_point_in_polygon(polygon: List[Point], margin: float = 5.0) -> Point:
        """Generate random point inside polygon with margin from edges."""
        min_x, max_x, min_y, max_y = GeometryUtils.get_polygon_bounds(polygon)
        
        # Add margin to bounds
        min_x += margin
        max_x -= margin
        min_y += margin
        max_y -= margin
        
        max_attempts = 1000
        for _ in range(max_attempts):
            point = Point(
                random.uniform(min_x, max_x),
                random.uniform(min_y, max_y)
            )
            if GeometryUtils.point_in_polygon(point, polygon):
                return point
        
        # Fallback: return center of polygon
        center_x = sum(p.x for p in polygon) / len(polygon)
        center_y = sum(p.y for p in polygon) / len(polygon)
        return Point(center_x, center_y)


class FoldOperations:
    """Handles folding operations and transformations."""
    
    FOLD_REFLECTIONS = {
        "V": lambda p: Point(120 - p.x, p.y),  # Vertical fold
        "H": lambda p: Point(p.x, 120 - p.y),  # Horizontal fold
        "D": lambda p: Point(p.y, p.x),        # Diagonal fold (y=x)
        "N": lambda p: Point(120 - p.y, 120 - p.x),  # Negative diagonal
    }
    
    @classmethod
    def apply_fold_to_point(cls, point: Point, fold: str) -> Point:
        """Apply fold transformation to a single point."""
        return cls.FOLD_REFLECTIONS[fold](point)
    
    @classmethod
    def get_fold_line(cls, polygon: List[Point], fold: str) -> Tuple[Point, Point]:
        """Get the fold line endpoints for visualization."""
        if not polygon:
            return Point(0, 0), Point(0, 0)
        
        min_x, max_x, min_y, max_y = GeometryUtils.get_polygon_bounds(polygon)
        
        if fold == "V":
            mid = (min_x + max_x) / 2
            return Point(mid, min_y - 5), Point(mid, max_y + 5)
        elif fold == "H":
            mid = (min_y + max_y) / 2
            return Point(min_x - 5, mid), Point(max_x + 5, mid)
        elif fold == "D":
            # Diagonal fold y = x + b, extend beyond polygon bounds
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            b = cy - cx
            return Point(min_x - 10, min_x - 10 + b), Point(max_x + 10, max_x + 10 + b)
        elif fold == "N":
            # Negative diagonal fold x + y = constant
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            c = cx + cy
            return Point(min_x - 10, c - (min_x - 10)), Point(max_x + 10, c - (max_x + 10))
        
        return Point(0, 0), Point(0, 0)
    
    @classmethod
    def compute_unfolded_holes(cls, holes: List[Point], folds: List[str]) -> List[Point]:
        """Compute all hole positions after unfolding."""
        layers = [holes]
        
        for fold in folds:
            new_layers = []
            for layer in layers:
                new_layers.append(layer)
                reflected_layer = [cls.apply_fold_to_point(hole, fold) for hole in layer]
                new_layers.append(reflected_layer)
            layers = new_layers
        
        # Flatten all layers
        return [hole for layer in layers for hole in layer]
    
    @classmethod
    def clip_polygon_by_fold(cls, polygon: List[Point], fold: str) -> List[Point]:
        """Clip polygon by fold line, keeping only the folded portion."""
        if not polygon:
            return []
        
        min_x, max_x, min_y, max_y = GeometryUtils.get_polygon_bounds(polygon)
        
        if fold == "V":
            mid = (min_x + max_x) / 2
            inside = lambda p: p.x >= mid
            intersect = lambda p1, p2: Point(mid, p1.y + (mid - p1.x) * (p2.y - p1.y) / (p2.x - p1.x))
        elif fold == "H":
            mid = (min_y + max_y) / 2
            inside = lambda p: p.y >= mid
            intersect = lambda p1, p2: Point(p1.x + (mid - p1.y) * (p2.x - p1.x) / (p2.y - p1.y), mid)
        elif fold == "D":
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            b = cy - cx
            inside = lambda p: p.y <= p.x + b
            intersect = lambda p1, p2: cls._diagonal_intersect(p1, p2, b)
        elif fold == "N":
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            inside = lambda p: (p.x + p.y) <= (cx + cy)
            intersect = lambda p1, p2: cls._negative_diagonal_intersect(p1, p2, cx + cy)
        else:
            return polygon
        
        return cls._sutherland_hodgman_clip(polygon, inside, intersect)
    
    @staticmethod
    def _diagonal_intersect(p1: Point, p2: Point, b: float) -> Point:
        """Compute intersection with diagonal line y = x + b."""
        denom = (p2.x - p1.x) - (p2.y - p1.y)
        if abs(denom) < 1e-10:
            return p1
        t = (p1.y - p1.x - b) / denom
        return Point(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
    
    @staticmethod
    def _negative_diagonal_intersect(p1: Point, p2: Point, sum_val: float) -> Point:
        """Compute intersection with negative diagonal line x + y = sum_val."""
        denom = (p2.x - p1.x) + (p2.y - p1.y)
        if abs(denom) < 1e-10:
            return p1
        t = (sum_val - (p1.x + p1.y)) / denom
        return Point(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
    
    @staticmethod
    def _sutherland_hodgman_clip(polygon: List[Point], inside_func, intersect_func) -> List[Point]:
        """Sutherland-Hodgman polygon clipping algorithm."""
        if not polygon:
            return []
        
        clipped = []
        prev_point = polygon[-1]
        
        for curr_point in polygon:
            if inside_func(curr_point):
                if not inside_func(prev_point):
                    clipped.append(intersect_func(prev_point, curr_point))
                clipped.append(curr_point)
            elif inside_func(prev_point):
                clipped.append(intersect_func(prev_point, curr_point))
            prev_point = curr_point
        
        return clipped


class PaperRenderer:
    """Handles rendering of paper folding visualizations using matplotlib."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def draw_fold_line(self, ax: plt.Axes, start: Point, end: Point):
        """Draw a dashed fold line."""
        ax.plot([start.x, end.x], [start.y, end.y], 
                'w--', linewidth=self.config.linewidth * 1.5, alpha=0.8, zorder=15)
        
    def create_subplot(self, fig, position, title: str = "") -> plt.Axes:
        """Create a subplot with paper background."""
        ax = fig.add_subplot(position)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 120)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Draw paper background
        paper_rect = patches.Rectangle(
            (self.config.paper_margin, self.config.paper_margin),
            self.config.paper_size, self.config.paper_size,
            linewidth=self.config.linewidth, edgecolor='black', facecolor='white'
        )
        ax.add_patch(paper_rect)
        
        return ax
    
    def draw_holes(self, ax: plt.Axes, holes: List[Point]):
        """Draw holes with smart positioning for overlapping holes."""
        hole_groups = self._group_nearby_holes(holes)
        
        for group in hole_groups:
            if len(group) == 1:
                hole = group[0]
                circle = patches.Circle(
                    (hole.x, hole.y), self.config.hole_radius,
                    color='black', zorder=10
                )
                ax.add_patch(circle)
            else:
                self._draw_clustered_holes(ax, group)
    
    def _group_nearby_holes(self, holes: List[Point], threshold: float = 4.0) -> List[List[Point]]:
        """Group holes that are close to each other."""
        groups = []
        used = set()
        
        for i, hole in enumerate(holes):
            if i in used:
                continue
            
            group = [hole]
            used.add(i)
            
            for j, other_hole in enumerate(holes[i+1:], i+1):
                if j in used:
                    continue
                
                distance = math.sqrt((hole.x - other_hole.x)**2 + (hole.y - other_hole.y)**2)
                if distance <= threshold:
                    group.append(other_hole)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _draw_clustered_holes(self, ax: plt.Axes, holes: List[Point]):
        """Draw clustered holes in a circular pattern."""
        if not holes:
            return
        
        center_x = sum(h.x for h in holes) / len(holes)
        center_y = sum(h.y for h in holes) / len(holes)
        
        radius_offset = 3.0
        for i, hole in enumerate(holes):
            angle = 2 * math.pi * i / len(holes)
            x = center_x + radius_offset * math.cos(angle)
            y = center_y + radius_offset * math.sin(angle)
            
            circle = patches.Circle(
                (x, y), self.config.hole_radius,
                color='black', zorder=10
            )
            ax.add_patch(circle)
    
    def draw_polygon(self, ax: plt.Axes, polygon: List[Point], 
                    facecolor: str = 'white', edgecolor: str = 'black'):
        """Draw a polygon on the axes."""
        if len(polygon) < 3:
            return
        
        coords = [(p.x, p.y) for p in polygon]
        poly_patch = patches.Polygon(
            coords, closed=True, 
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=self.config.linewidth, zorder=5
        )
        ax.add_patch(poly_patch)


class WrongOptionGenerator:
    """Generates wrong answer options for the paper folding test."""
    
    @staticmethod
    def remove_random_hole(holes: List[Point]) -> List[Point]:
        """Remove a random hole or slightly move it."""
        if not holes:
            return holes
        
        result = holes.copy()
        if len(result) > 1:
            idx = random.randrange(len(result))
            result.pop(idx)
        elif len(result) == 1:
            hole = result[0]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            result[0] = Point(hole.x + dx, hole.y + dy)
        
        return result
    
    @staticmethod
    def mirror_holes(holes: List[Point]) -> List[Point]:
        """Mirror holes across center point."""
        return [Point(120 - h.x, 120 - h.y) for h in holes]
    
    @staticmethod
    def rotate_holes(holes: List[Point]) -> List[Point]:
        """Rotate holes 90 degrees around center."""
        result = []
        for hole in holes:
            # Translate to origin, rotate, translate back
            tx, ty = hole.x - 60, hole.y - 60
            rx, ry = ty, -tx
            result.append(Point(rx + 60, ry + 60))
        return result


class PaperFoldingGenerator:
    """Main class for generating paper folding test images."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.renderer = PaperRenderer(self.config)
        self.wrong_generator = WrongOptionGenerator()
        
    def generate_test_image(self, folds: List[str], test_number: int, 
                          num_holes: int) -> Tuple[str, str]:
        """Generate a complete test image with folding process and options."""
        
        # Generate folding process
        fold_images = self._generate_fold_process(folds, num_holes)
        
        # Generate answer options
        correct_holes, wrong_options = self._generate_answer_options(
            fold_images[-1]['holes'], folds
        )
        
        # Create final composite image
        return self._create_composite_image(
            fold_images, correct_holes, wrong_options, 
            test_number, len(folds), num_holes
        )
    
    def _generate_fold_process(self, folds: List[str], num_holes: int) -> List[Dict]:
        """Generate the step-by-step folding process."""
        process_steps = []
        
        # Initial unfolded paper
        initial_polygon = [
            Point(self.config.paper_margin, self.config.paper_margin),
            Point(self.config.paper_margin + self.config.paper_size, self.config.paper_margin),
            Point(self.config.paper_margin + self.config.paper_size, 
                  self.config.paper_margin + self.config.paper_size),
            Point(self.config.paper_margin, self.config.paper_margin + self.config.paper_size)
        ]
        
        process_steps.append({
            'polygon': initial_polygon,
            'title': 'Unfolded',
            'holes': [],
            'fold_line': None
        })
        
        # Apply each fold
        current_polygon = initial_polygon
        for i, fold in enumerate(folds):
            # Get fold line before clipping
            fold_line = FoldOperations.get_fold_line(current_polygon, fold)
            current_polygon = FoldOperations.clip_polygon_by_fold(current_polygon, fold)
            process_steps.append({
                'polygon': current_polygon,
                'title': f'Fold {i + 1}',
                'holes': [],
                'fold_line': fold_line
            })
        
        # Add holes to final folded state
        if current_polygon:
            holes = [
                GeometryUtils.generate_point_in_polygon(
                    current_polygon, self.config.hole_margin
                ) for _ in range(num_holes)
            ]
            process_steps[-1]['holes'] = holes
            process_steps.append({
                'polygon': current_polygon,
                'title': 'Final view',
                'holes': holes,
                'fold_line': None
            })
        
        return process_steps
    
    def _generate_answer_options(self, punched_holes: List[Point], 
                               folds: List[str]) -> Tuple[List[Point], List[List[Point]]]:
        """Generate correct answer and wrong options."""
        # Correct answer: unfold the holes
        correct_holes = FoldOperations.compute_unfolded_holes(punched_holes, folds)
        
        # Generate wrong options
        wrong_option_1 = self.wrong_generator.remove_random_hole(correct_holes)
        wrong_option_2 = self.wrong_generator.mirror_holes(correct_holes)
        
        # Ensure wrong options are actually different
        if self._holes_are_equivalent(wrong_option_2, correct_holes):
            wrong_option_2 = self.wrong_generator.rotate_holes(correct_holes)
            if self._holes_are_equivalent(wrong_option_2, correct_holes):
                wrong_option_2 = [Point(min(h.x + 5, 110), h.y) for h in correct_holes]
        
        return correct_holes, [wrong_option_1, wrong_option_2]
    
    def _holes_are_equivalent(self, holes1: List[Point], holes2: List[Point]) -> bool:
        """Check if two hole sets are equivalent (same positions)."""
        if len(holes1) != len(holes2):
            return False
        
        set1 = {(round(h.x, 1), round(h.y, 1)) for h in holes1}
        set2 = {(round(h.x, 1), round(h.y, 1)) for h in holes2}
        return set1 == set2
    
    def _create_composite_image(self, fold_images: List[Dict], 
                              correct_holes: List[Point], wrong_options: List[List[Point]],
                              test_number: int, num_folds: int, num_holes: int) -> Tuple[str, str]:
        """Create the final composite image with all steps and options."""
        
        # Calculate layout
        num_process_steps = len(fold_images)
        process_width = num_process_steps * 2 + (num_process_steps - 1) * 0.5
        options_width = 3 * 2 + 2 * 0.5
        total_width = max(process_width, options_width)
        
        fig = plt.figure(figsize=(total_width * 1.2, 8))
        
        # Draw process steps (top row)
        process_start = (total_width - process_width) / 2
        for i, step in enumerate(fold_images):
            left = process_start + i * 2.5
            ax = plt.subplot2grid((4, int(total_width * 2)), (0, int(left * 2)), 
                                colspan=4, rowspan=2)
            ax = self._setup_subplot(ax, step['title'])
            
            if step['polygon']:
                # Always draw paper as white
                self.renderer.draw_polygon(ax, step['polygon'], 'white', 'black')
            
            # Draw fold line for all fold steps (including the previous step showing where to fold)
            if step.get('fold_line'):
                fold_start, fold_end = step['fold_line']
                self.renderer.draw_fold_line(ax, fold_start, fold_end)
            
            if step['holes']:
                self.renderer.draw_holes(ax, step['holes'])
        
        # Draw answer options (bottom row)
        options = [
            ("correct", correct_holes),
            ("wrong", wrong_options[0]),
            ("wrong", wrong_options[1])
        ]
        random.shuffle(options)
        
        correct_label = None
        option_labels = ["Option A", "Option B", "Option C"]
        options_start = (total_width - options_width) / 2
        
        for i, (option_type, holes) in enumerate(options):
            left = options_start + i * 2.5
            ax = plt.subplot2grid((4, int(total_width * 2)), (2, int(left * 2)), 
                                colspan=4, rowspan=2)
            ax = self._setup_subplot(ax, option_labels[i])
            
            if option_type == "correct":
                correct_label = option_labels[i]
            
            self.renderer.draw_holes(ax, holes)
        
        # Save image
        filename = f"{test_number}_fold-{num_folds}_holes-{num_holes}.png"
        filepath = os.path.join("data/pf", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return filepath, correct_label
    
    def _setup_subplot(self, ax: plt.Axes, title: str) -> plt.Axes:
        """Setup a subplot with consistent formatting."""
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 120)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Only draw paper background for answer options, not for fold steps
        if title in ['Option A', 'Option B', 'Option C']:
            paper_rect = patches.Rectangle(
                (self.config.paper_margin, self.config.paper_margin),
                self.config.paper_size, self.config.paper_size,
                linewidth=self.config.linewidth, edgecolor='black', facecolor='white'
            )
            ax.add_patch(paper_rect)
        
        return ax


def main():
    """Main entry point for the paper folding generator."""
    parser = argparse.ArgumentParser(
        description="Generate paper folding test images using matplotlib."
    )
    parser.add_argument(
        "-n", "--num-images", type=int, default=10,
        help="Number of images to generate (default: 10)"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "-f", "--num-folds", type=int, default=2, choices=range(1, 10),
        help="Number of folds (1-9, default: 2)"
    )
    parser.add_argument(
        "-H", "--num-holes", type=int, default=1,
        help="Number of holes to punch (default: 1)"
    )
    parser.add_argument(
        "-m", "--metadata-file", type=str, default="metadata.jsonl",
        help="Metadata file name (default: metadata.jsonl)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="data/pf",
        help="Output directory (default: data/pf)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    config = Config()
    generator = PaperFoldingGenerator(config)
    
    # Generate images
    metadata_path = os.path.join(args.output_dir, args.metadata_file)
    with open(metadata_path, "a", encoding="utf-8") as metaf:
        for i in range(1, args.num_images + 1):
            # Choose fold type
            fold_group = random.choice(["VH", "Diagonal"])
            if fold_group == "VH":
                folds = [random.choice(["V", "H"]) for _ in range(args.num_folds)]
            else:
                folds = [random.choice(["D", "N"]) for _ in range(args.num_folds)]
            
            # Generate test image
            image_path, correct_option = generator.generate_test_image(
                folds, i, args.num_holes
            )
            
            # Save metadata
            metadata = {
                "filename": os.path.basename(image_path),
                "correct_option": correct_option,
                "folds": folds,
                "num_holes": args.num_holes,
            }
            metaf.write(json.dumps(metadata) + "\n")
    
    print(f"Generated {args.num_images} paper folding test images in {args.output_dir}")


if __name__ == "__main__":
    main()
