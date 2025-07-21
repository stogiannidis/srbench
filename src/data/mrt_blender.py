#!/usr/bin/env python3

import bpy
import bmesh
import os
import sys
import random
import argparse
import json
import mathutils
from mathutils import Vector, Matrix, Euler
import math

# Import shape definitions from mrt.py
sys.path.append(os.path.dirname(__file__))
from mrt import SHAPES, EASY_SHAPES, COMPLEX_SHAPES, SIMILAR_MAPPING

def clear_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_cube_at_position(position, size=1.0, name="Cube"):
    """Create a cube at the specified position."""
    bpy.ops.mesh.primitive_cube_add(size=size, location=position)
    cube = bpy.context.active_object
    cube.name = name
    return cube

def create_polycube(shape_coords, cube_size=1.0, material=None):
    """Create a polycube from coordinate list."""
    cubes = []
    for i, coord in enumerate(shape_coords):
        pos = Vector((coord[0] * cube_size, coord[1] * cube_size, coord[2] * cube_size))
        cube = create_cube_at_position(pos, cube_size, f"Cube_{i}")
        if material:
            cube.data.materials.append(material)
        cubes.append(cube)
    
    # Join all cubes into one object
    bpy.ops.object.select_all(action='DESELECT')
    for cube in cubes:
        cube.select_set(True)
    bpy.context.view_layer.objects.active = cubes[0]
    bpy.ops.object.join()
    
    polycube = bpy.context.active_object
    return polycube

def create_material(color=(1, 1, 1, 1), name="Material"):
    """Create a material with specified color."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = color  # Base Color
    bsdf.inputs[7].default_value = 0.2   # Roughness
    bsdf.inputs[15].default_value = 1.0  # Specular
    return mat

def setup_lighting():
    """Set up clean lighting for the scene."""
    # Add sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (0.785, 0, 0.785)  # 45 degrees
    
    # Add fill light
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 8))
    fill = bpy.context.active_object
    fill.data.energy = 1.0
    fill.data.size = 5.0

def setup_camera(target_location, distance=8.0):
    """Set up camera to look at target location."""
    # Add camera
    bpy.ops.object.camera_add(location=(distance, -distance, distance))
    camera = bpy.context.active_object
    
    # Point camera at target
    direction = target_location - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    # Set as active camera
    bpy.context.scene.camera = camera
    return camera

def get_object_bounds(obj):
    """Get the bounding box of an object."""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_coord = Vector((min(v.x for v in bbox_corners),
                       min(v.y for v in bbox_corners), 
                       min(v.z for v in bbox_corners)))
    max_coord = Vector((max(v.x for v in bbox_corners),
                       max(v.y for v in bbox_corners),
                       max(v.z for v in bbox_corners)))
    center = (min_coord + max_coord) / 2
    size = max_coord - min_coord
    return center, size

def transform_rotate_blender(obj, difficulty="easy"):
    """Apply rotation transformation to object."""
    if difficulty == "easy":
        # Single axis rotation with simple angles
        axis = random.choice(['X', 'Y', 'Z'])
        angle = math.radians(random.choice([-90, 90, 180]))
        
        if axis == 'X':
            obj.rotation_euler = (angle, 0, 0)
        elif axis == 'Y':
            obj.rotation_euler = (0, angle, 0)
        else:  # Z
            obj.rotation_euler = (0, 0, angle)
    else:  # complex
        # Multi-axis rotation
        angle_x = math.radians(random.choice([0, 60, 90, 120]))
        angle_y = math.radians(random.choice([0, 60, 90, 120]))
        angle_z = math.radians(random.choice([0, 60, 90, 120]))
        obj.rotation_euler = (angle_x, angle_y, angle_z)

def transform_mirror_blender(obj, difficulty="easy"):
    """Apply mirror transformation to object."""
    if difficulty == "easy":
        # Mirror across Z axis
        obj.scale[2] = -1
    else:  # complex
        # Mirror across random axis
        axis = random.choice([0, 1, 2])
        scale = [1, 1, 1]
        scale[axis] = -1
        obj.scale = scale

def render_image(output_path, resolution=(512, 512)):
    """Render the current scene to an image."""
    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'PNG'
    
    # Set render engine to Cycles for better quality
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    
    bpy.ops.render.render(write_still=True)

def create_composite_image(original_obj, candidates, output_path, difficulty="easy"):
    """Create composite image with original and candidates."""
    # Position objects for composite layout
    if difficulty == "easy":
        positions = [
            Vector((0, 0, 4)),    # Original (top center)
            Vector((-4, 0, 0)),   # Candidate A
            Vector((0, 0, 0)),    # Candidate B  
            Vector((4, 0, 0)),    # Candidate C
        ]
    else:  # complex
        positions = [
            Vector((0, 0, 6)),    # Original (top center)
            Vector((-6, 0, 0)),   # Candidate A
            Vector((-2, 0, 0)),   # Candidate B
            Vector((2, 0, 0)),    # Candidate C
            Vector((6, 0, 0)),    # Candidate D
        ]
    
    # Position original
    original_obj.location = positions[0]
    
    # Position candidates
    for i, (_, candidate_obj) in enumerate(candidates):
        candidate_obj.location = positions[i + 1]
    
    # Set up camera to capture all objects
    all_objects = [original_obj] + [obj for _, obj in candidates]
    
    # Calculate scene bounds
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for obj in all_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_coords.x = min(min_coords.x, world_corner.x)
            min_coords.y = min(min_coords.y, world_corner.y)
            min_coords.z = min(min_coords.z, world_corner.z)
            max_coords.x = max(max_coords.x, world_corner.x)
            max_coords.y = max(max_coords.y, world_corner.y)
            max_coords.z = max(max_coords.z, world_corner.z)
    
    scene_center = (min_coords + max_coords) / 2
    scene_size = max(max_coords - min_coords)
    
    # Position camera
    camera_distance = scene_size * 1.5
    setup_camera(scene_center, camera_distance)
    
    # Render
    render_image(output_path)

def generate_one_image_blender(index, difficulty="easy", facecolor="white", outdir="data/mrt"):
    """Generate a single MRT image using Blender."""
    clear_scene()
    
    # Create material
    if facecolor == "white":
        color = (0.9, 0.9, 0.9, 1.0)
    else:
        # Simple color mapping - extend as needed
        color_map = {
            "red": (0.8, 0.2, 0.2, 1.0),
            "blue": (0.2, 0.2, 0.8, 1.0),
            "green": (0.2, 0.8, 0.2, 1.0),
        }
        color = color_map.get(facecolor, (0.9, 0.9, 0.9, 1.0))
    
    material = create_material(color, "PolycubeMaterial")
    
    # Select shapes based on difficulty
    shapes_list = EASY_SHAPES if difficulty == "easy" else COMPLEX_SHAPES
    shape_name = random.choice(shapes_list)
    shape_coords = SHAPES[shape_name]
    
    # Create original polycube
    original_obj = create_polycube(shape_coords, material=material)
    original_obj.name = "Original"
    
    # Generate candidates
    candidates = []
    
    # Correct candidate (rotation)
    correct_obj = create_polycube(shape_coords, material=material)
    correct_obj.name = "Rotate"
    transform_rotate_blender(correct_obj, difficulty)
    candidates.append(("rotate", correct_obj))
    
    # Mirror candidate
    mirror_obj = create_polycube(shape_coords, material=material)
    mirror_obj.name = "Mirror"
    transform_mirror_blender(mirror_obj, difficulty)
    transform_rotate_blender(mirror_obj, difficulty)
    candidates.append(("mirror", mirror_obj))
    
    # Similar shape candidate
    if shape_name in SIMILAR_MAPPING:
        similar_candidates = SIMILAR_MAPPING[shape_name][:]
        random.shuffle(similar_candidates)
        similar_shape_name = similar_candidates[0] if similar_candidates else shape_name
    else:
        similar_shape_name = shape_name
    
    similar_coords = SHAPES[similar_shape_name]
    similar_obj = create_polycube(similar_coords, material=material)
    similar_obj.name = "Similar"
    transform_rotate_blender(similar_obj, difficulty)
    candidates.append(("similar", similar_obj))
    
    # Add second mirror for complex mode
    if difficulty == "complex":
        mirror2_obj = create_polycube(shape_coords, material=material)
        mirror2_obj.name = "Mirror2"
        transform_mirror_blender(mirror2_obj, difficulty)
        transform_rotate_blender(mirror2_obj, difficulty)
        candidates.append(("mirror2", mirror2_obj))
    
    # Shuffle candidates
    random.shuffle(candidates)
    correct_candidate_index = [
        i for i, cand in enumerate(candidates) if cand[0] == "rotate"
    ][0]
    
    # Set up lighting
    setup_lighting()
    
    # Create composite image
    filename = f"{shape_name}_{index}.png"
    output_path = os.path.join(outdir, filename)
    create_composite_image(original_obj, candidates, output_path, difficulty)
    
    # Save metadata
    metadata = {
        "filename": filename,
        "difficulty": difficulty,
        "shape": shape_name,
        "candidate_order": [tag for tag, _ in candidates],
        "answer": chr(65 + correct_candidate_index),
    }
    
    metadata_path = os.path.join(outdir, "metadata.jsonl")
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate mental rotation test images using Blender."
    )
    parser.add_argument(
        "--difficulty", "-d", type=str, choices=["easy", "complex"], default="easy",
        help="Difficulty level: 'easy' or 'complex'"
    )
    parser.add_argument(
        "--num_images", "-n", type=int, default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--color", "-c", type=str, default="white",
        help="Color for the polycubes"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=69,
        help="Seed for reproducible results"
    )
    parser.add_argument(
        "--outdir", "-o", type=str, default=None,
        help="Output directory (defaults to data/mrt_blender/{difficulty})"
    )
    
    args = parser.parse_args()
    
    # Set default output directory based on difficulty
    if args.outdir is None:
        args.outdir = f"data/mrt_blender/{args.difficulty}"
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Set seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
    
    # Generate images
    for i in range(args.num_images):
        generate_one_image_blender(i, difficulty=args.difficulty, 
                                 facecolor=args.color, outdir=args.outdir)
    
    print(f"Generated {args.num_images} {args.difficulty} MRT images in {args.outdir}")

# Run directly in Blender
if __name__ == "__main__":
    # When running in Blender, parse sys.argv differently
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
        sys.argv = [sys.argv[0]] + argv
    main()
