# ------------------------------------------------------------
# unified_path_generator.py
# ------------------------------------------------------------
import os
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

"""
Program that generates shapes.
Shapes will be generated according to the classes list(line 156).
Example: dict(shape="circle", params=dict(center=(500, 500), radius=50 ),  text="small circle in the middle", n=2000)

Arguments:	
	out_file: name of the output dataset file (.pt)
"""

# Generic helpers

def move_straight_line(start, end, step_size=50):
    start, end = np.asarray(start), np.asarray(end)
    dist = np.linalg.norm(end - start)
    if dist == 0:
        return np.array([start])
    direction = (end - start) / dist
    n_steps = int(dist // step_size)
    pts = [start + i * step_size * direction for i in range(n_steps + 1)]
    pts.append(end)
    return np.array(pts)

def add_noise(points: np.ndarray, noise_level=0.05):
    return points + np.random.normal(scale=noise_level, size=points.shape)

# Circle utilities

def gen_circle_outline(cx, cy, r, n=40, phase=None):
    if phase is None:
        phase = np.random.uniform(0, 2 * np.pi)
    angles = np.linspace(phase, phase + 2 * np.pi, n)
    return np.stack([cx + r * np.cos(angles),
                     cy + r * np.sin(angles)], axis=1)

def closest_on_circle(cx, cy, r, pt):
    outline = gen_circle_outline(cx, cy, r)
    idx = np.argmin(np.linalg.norm(outline - pt, axis=1))
    return outline[idx]

def gen_circle_trace(cx, cy, r, start_pt, n=40):
    start_angle = np.arctan2(start_pt[1] - cy, start_pt[0] - cx)
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, n)
    return np.stack([cx + r * np.cos(angles),
                     cy + r * np.sin(angles)], axis=1)

#square utilites
def square_outline(cx, cy, size, pts_per_side=20):
    h = size / 2
    tl = (cx - h, cy - h); tr = (cx + h, cy - h)
    br = (cx + h, cy + h); bl = (cx - h, cy + h)
    top    = np.linspace(tl, tr, pts_per_side)
    right  = np.linspace(tr, br, pts_per_side)
    bottom = np.linspace(br, bl, pts_per_side)
    left   = np.linspace(bl, tl, pts_per_side)
    return np.vstack([top, right, bottom, left])

def closest_on_square(points, pt):
    d = np.linalg.norm(points - pt, axis=1)
    return points[np.argmin(d)]

def roll_to_start(points, start_pt):
    idx = np.argmin(np.linalg.norm(points - start_pt, axis=1))
    return np.roll(points, -idx, axis=0)

# One-class generator

def build_paths(
        shape_type: str,
        shape_params: Dict,
        text: str,
        num_origins: int,
        density: float = 0.1  # Controls how dense the points are based on perimeter
) -> List[Dict]:
    """
    Returns a list[dict] each with keys {'path': Tensor, 'text': str}
    Automatically adjusts number of points based on shape size.
    """
    out = []
    for _ in range(num_origins):
        start_pt = (random.randint(0, 1000), random.randint(0, 1000))

        if shape_type == "circle":
            cx, cy, r = shape_params["center"][0], shape_params["center"][1], shape_params["radius"]
            circumference = 2 * np.pi * r
            n_points = max(5, int(circumference * density))  # ensure at least 10
            closest = closest_on_circle(cx, cy, r, start_pt)
            to_shape = move_straight_line(start_pt, closest)
            perimeter = gen_circle_trace(cx, cy, r, closest, n=n_points)

        elif shape_type == "square":
            cx, cy, size = shape_params["center"][0], shape_params["center"][1], shape_params["size"]
            perimeter_length = 4 * size
            pts_per_side = max(2, int((perimeter_length * density) // 4))  # ensure at least 5/side
            outline = square_outline(cx, cy, size, pts_per_side=pts_per_side)
            closest = closest_on_square(outline, start_pt)
            to_shape = move_straight_line(start_pt, closest)
            perimeter = roll_to_start(outline, closest)

        else:
            raise ValueError(f"Unknown shape_type '{shape_type}'")

        path_lst = []
        for x, y in to_shape:
            path_lst.append([x, y, 0, 0])  # move
        for i in range(len(perimeter) - 1):
            x1, y1 = perimeter[i]
            x2, y2 = perimeter[i + 1]
            path_lst.append([x1, y1, 1, 0])  # draw
            path_lst.append([x2, y2, 1, 0])

        path_lst[-1][-1] = 1  # stop flag

        path_tensor = torch.tensor(path_lst, dtype=torch.float32).clone().detach()
        path_tensor[:, :2] = torch.tensor(add_noise(path_tensor[:, :2]), dtype=torch.float32).clone().detach()

        out.append({"path": path_tensor, "text": text})
    return out


# ------------------------------------------------------------
# ----------  Incremental saver
# ------------------------------------------------------------
def append_to_pt(data_chunk: List[Dict], out_file: str):
    if os.path.exists(out_file):
        prev = torch.load(out_file)
        prev.extend(data_chunk)
        torch.save(prev, out_file)
    else:
        torch.save(data_chunk, out_file)
    print(f"  ↳ wrote {len(data_chunk)} items  →  {out_file} [total: {len(torch.load(out_file))}]")

# ------------------------------------------------------------
# ----------  Visual sanity-check (optional)
# ------------------------------------------------------------
def quick_plot(sample, title="sample"):
    xs, ys, acts = sample["path"][:,0], sample["path"][:,1], sample["path"][:,2]
    move = acts < 0.5; draw = acts >= 0.5
    plt.figure(figsize=(4,4)); plt.scatter(xs[move], ys[move], c='blue', s=6)
    plt.scatter(xs[draw], ys[draw], c='red', s=6); plt.gca().set_aspect('equal')
    plt.title(title); plt.show()

# ------------------------------------------------------------
# ----------  Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Generate mixed shape path dataset")
    parser.add_argument("--out_file", default="mixed_paths.pt")
    parser.add_argument("--plot", action="store_true", help="Show a sample plot")
    parser.add_argument("--density", type=float, default=0.1, help = "How dense will path be")
    parser.add_argument("--n", type=int, default = 100, help="How many paths per annotation to generate")
    args = parser.parse_args()
    n = args.n
    density = args.density    

    # ---------------------------------------------------------------------
    # Describe each class you want to generate
    # ---------------------------------------------------------------------

    
    classes = [
    # ----------------- CIRCLES -----------------
    dict(shape="circle", params=dict(center=(500, 500), radius=50 ),  text="small circle in the middle",         n=n),
    dict(shape="circle", params=dict(center=(500, 500), radius=100),  text="medium circle in the middle",        n=n),
    dict(shape="circle", params=dict(center=(500, 500), radius=150),  text="big circle in the middle",           n=n),

    dict(shape="circle", params=dict(center=(500, 200), radius=50 ),  text="small circle on the bottom side",    n=n),
    dict(shape="circle", params=dict(center=(500, 200), radius=100),  text="medium circle on the bottom side",   n=n),
    dict(shape="circle", params=dict(center=(500, 200), radius=150),  text="big circle on the bottom side",      n=n),

    dict(shape="circle", params=dict(center=(200, 500), radius=50 ),  text="small circle on the left side",      n=n),
    dict(shape="circle", params=dict(center=(200, 500), radius=100),  text="medium circle on the left side",     n=n),
    dict(shape="circle", params=dict(center=(200, 500), radius=150),  text="big circle on the left side",        n=n),

    dict(shape="circle", params=dict(center=(800, 500), radius=50 ),  text="small circle on the right side",     n=n),
    dict(shape="circle", params=dict(center=(800, 500), radius=100),  text="medium circle on the right side",    n=n),
    dict(shape="circle", params=dict(center=(800, 500), radius=150),  text="big circle on the right side",       n=n),

    dict(shape="circle", params=dict(center=(500, 800), radius=50 ),  text="small circle on the top side",       n=n),
    dict(shape="circle", params=dict(center=(500, 800), radius=100),  text="medium circle on the top side",      n=n),
    dict(shape="circle", params=dict(center=(500, 800), radius=150),  text="big circle on the top side",         n=n),
    
    dict(shape="circle", params=dict(center=(800, 800), radius=50 ),  text="small circle in the top right corner",  n=n),
    dict(shape="circle", params=dict(center=(800, 800), radius=100),  text="medium circle in the top right corner", n=n),
    dict(shape="circle", params=dict(center=(800, 800), radius=150),  text="big circle in the top right corner",    n=n),

    dict(shape="circle", params=dict(center=(200, 800), radius=50 ),  text="small circle in the top left corner",   n=n),
    dict(shape="circle", params=dict(center=(200, 800), radius=100),  text="medium circle in the top left corner",  n=n),
    dict(shape="circle", params=dict(center=(200, 800), radius=150),  text="big circle in the top left corner",     n=n),

    dict(shape="circle", params=dict(center=(200, 200), radius=50 ),  text="small circle in the bottom left corner",  n=n),
    dict(shape="circle", params=dict(center=(200, 200), radius=100),  text="medium circle in the bottom left corner", n=n),
    dict(shape="circle", params=dict(center=(200, 200), radius=150),  text="big circle in the bottom left corner",    n=n),

    dict(shape="circle", params=dict(center=(800, 200), radius=50 ),  text="small circle in the bottom right corner",  n=n),
    dict(shape="circle", params=dict(center=(800, 200), radius=100),  text="medium circle in the bottom right corner", n=n),
    dict(shape="circle", params=dict(center=(800, 200), radius=150),  text="big circle in the bottom right corner",    n=n),

    # ----------------- SQUARES -----------------
    dict(shape="square", params=dict(center=(500, 500), size=100 ),  text="small square in the middle",         n=n),
    dict(shape="square", params=dict(center=(500, 500), size=200 ),  text="medium square in the middle",        n=n),
    dict(shape="square", params=dict(center=(500, 500), size=300 ),  text="big square in the middle",           n=n),

    dict(shape="square", params=dict(center=(500, 200), size=100 ),  text="small square on the bottom side",    n=n),
    dict(shape="square", params=dict(center=(500, 200), size=200 ),  text="medium square on the bottom side",   n=n),
    dict(shape="square", params=dict(center=(500, 200), size=300 ),  text="big square on the bottom side",      n=n),

    dict(shape="square", params=dict(center=(200, 500), size=100 ),  text="small square on the left side",      n=n),
    dict(shape="square", params=dict(center=(200, 500), size=200 ),  text="medium square on the left side",     n=n),
    dict(shape="square", params=dict(center=(200, 500), size=300 ),  text="big square on the left side",        n=n),

    dict(shape="square", params=dict(center=(800, 500), size=100 ),  text="small square on the right side",     n=n),
    dict(shape="square", params=dict(center=(800, 500), size=200 ),  text="medium square on the right side",    n=n),
    dict(shape="square", params=dict(center=(800, 500), size=300 ),  text="big square on the right side",       n=n),

    dict(shape="square", params=dict(center=(500, 800), size=100 ),  text="small square on the top side",       n=n),
    dict(shape="square", params=dict(center=(500, 800), size=200 ),  text="medium square on the top side",      n=n),
    dict(shape="square", params=dict(center=(500, 800), size=300 ),  text="big square on the top side",         n=n),

    dict(shape="square", params=dict(center=(800, 800), size=100 ),  text="small square in the top right corner",  n=n),
    dict(shape="square", params=dict(center=(800, 800), size=200 ),  text="medium square in the top right corner", n=n),
    dict(shape="square", params=dict(center=(800, 800), size=300 ),  text="big square in the top right corner",    n=n),

    dict(shape="square", params=dict(center=(200, 800), size=100 ),  text="small square in the top left corner",   n=n),
    dict(shape="square", params=dict(center=(200, 800), size=200 ),  text="medium square in the top left corner",  n=n),
    dict(shape="square", params=dict(center=(200, 800), size=300 ),  text="big square in the top left corner",     n=n),

    dict(shape="square", params=dict(center=(200, 200), size=100 ),  text="small square in the bottom left corner",  n=n),
    dict(shape="square", params=dict(center=(200, 200), size=200 ),  text="medium square in the bottom left corner", n=n),
    dict(shape="square", params=dict(center=(200, 200), size=300 ),  text="big square in the bottom left corner",    n=n),

    dict(shape="square", params=dict(center=(800, 200), size=100 ),  text="small square in the bottom right corner",  n=n),
    dict(shape="square", params=dict(center=(800, 200), size=200 ),  text="medium square in the bottom right corner", n=n),
    dict(shape="square", params=dict(center=(800, 200), size=300 ),  text="big square in the bottom right corner",    n=n),
    
    ]

    print(f"Generating dataset → {args.out_file}")
    i = 0
    for cls in classes:
        i += 1
        print(f"• {cls['text']}  ({cls['shape']} × {cls['n']})")
        print(f"• Written {i}/{len(classes)}")
        chunk = build_paths(cls["shape"], cls["params"], cls["text"], cls["n"], density)
        append_to_pt(chunk, args.out_file)

    if args.plot:
        data = torch.load(args.out_file)
        quick_plot(random.choice(data))

if __name__ == "__main__":
    main()

