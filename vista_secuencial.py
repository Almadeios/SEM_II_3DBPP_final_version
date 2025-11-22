# vista_ga.py
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pyrender
import trimesh


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("buffer-size debe ser >= 1")
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(description="Visualizador de soluciones 3D.")
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Ruta completa al JSON a mostrar. Si no se entrega, se arma desde dataset/buffer.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET_NAME", "blockout"),
        help="Dataset dentro de ./dataset (por defecto blockout).",
    )
    parser.add_argument(
        "--buffer-size",
        type=positive_int,
        default=6,
        help="Valor K para buscar en resultados/<dataset>/.",
    )
    parser.add_argument(
        "--solution-prefix",
        default=None,
        help="Prefijo opcional (metodo) dentro de resultados/<dataset>/solucion_<prefijo>_kK.json.",
    )
    return parser.parse_args()


args = parse_args()
DATASET_ROOT = Path("dataset")
dataset_name = args.dataset

if args.json_path:
    result_path = Path(args.json_path)
    parts = result_path.parts
    if "resultados" in parts:
        idx = parts.index("resultados")
        if idx + 1 < len(parts):
            dataset_name = parts[idx + 1]
    INPUT_JSON = str(result_path)
else:
    result_dir = Path("resultados") / dataset_name
    prefix = args.solution_prefix or args.dataset
    base_name = f"solucion_{prefix}_k{args.buffer_size}"
    candidates = sorted(result_dir.glob(f"{base_name}_s*.json"))
    if candidates:
        INPUT_JSON = str(candidates[-1])
    else:
        INPUT_JSON = str(result_dir / f"{base_name}.json")

OBJ_DIR = DATASET_ROOT / dataset_name / "shape_vhacd"

if not os.path.exists(INPUT_JSON):
    raise FileNotFoundError(f"No se encontro el archivo de solucion: {INPUT_JSON}")

# Cargar objetos colocados
with open(INPUT_JSON) as f:
    placed = json.load(f)

# Crear escena de pyrender
scene = pyrender.Scene()


# Cargar dimensiones del contenedor si existe metadata
meta_path = os.path.splitext(INPUT_JSON)[0] + "_meta.json"
if os.path.exists(meta_path):
    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    container_dims = np.array(metadata.get("container_dims_mm", [320, 320, 300])) / 1000.0
else:
    container_dims = np.array([320, 320, 300]) / 1000.0

# Contenedor transparente
container_box = trimesh.creation.box(extents=container_dims)
container_box.visual.face_colors = [150, 200, 255, 40]
container_tf = np.eye(4)
container_tf[:3, 3] = container_dims / 2.0  # Centrado
scene.add(pyrender.Mesh.from_trimesh(container_box, smooth=False), pose=container_tf)

# Agregar objetos
for item in placed:
    obj_path = OBJ_DIR / Path(item["id"])
    mesh = trimesh.load(str(obj_path), force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    # Color aleatorio suave
    color = [random.randint(150, 255) for _ in range(3)] + [255]
    mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))

    # Posición + rotación guardada en el json si existe
    if "transform_matrix" in item:
        tf = np.array(item["transform_matrix"]).reshape(4, 4)
    else:
        tf = np.eye(4)
        tf[:3, 3] = item["position_m"]

    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=tf)

# Calcular volumen total del contenedor
volumen_total = np.prod(container_dims)

# Calcular volumen ocupado por los objetos colocados
volumen_ocupado = 0.0
# Calcular volumen ocupado por los objetos colocados
volumen_ocupado = 0.0
for item in placed:
    mesh = trimesh.load(str(OBJ_DIR / Path(item["id"])), force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()
    volumen_ocupado += mesh.volume

# resultados
porcentaje_ocupado = (volumen_ocupado / volumen_total) * 100
print(f"\n Objetos colocados: {len(placed)}")
print(f"Volumen contenedor: {volumen_total:.6f} m^3")
print(f"Volumen usado:     {volumen_ocupado:.6f} m^3")
print(f"Porcentaje lleno:  {porcentaje_ocupado:.2f}%")

# Mostrar
pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
