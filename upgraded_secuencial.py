# ------------------------------------------------------------
# 3DBPP - Heurística secuencial mejorada con buffer y caída
# Autor: Diego Augusto Villalobos Ruiz
# Curso: Seminario de Investigación II
#
# Origen y atribuciones:
# - Dataset y rutas: basadas en Zhao et al. (2023), que proveen id2shape.pt,
#   test_sequence.pt y mallas con VHACD en 'shape_vhacd'.
# - Colisiones y mallas: se usa la librería trimesh y su CollisionManager.
# - Orientaciones estables: se invoca trimesh.poses.compute_stable_poses;
#   es una función de TRIMESH. El fallback de rotaciones ortogonales es implementación propia.
# - Buffer de candidatos: inspirado en el "candidate buffer K" de Zhao et al. (2023),
#   adaptado aquí como una ventana local (BUFFER_SIZE) con selección codiciosa por score.
# - Caída por gravedad: implementación propia mediante barrido en Z hasta primera
#   postura no colisionante; aproxima asentamiento sin física dinámica.
# - Heightmap 2D y soporte: heurística propia para estimar apoyo superficial.
# - Criterio de score (z, delta_h, -support_frac, rough): diseño propio.
#
# Notas:
# - Este archivo integra librerías de terceros (trimesh, numpy, torch, tqdm)
#   bajo sus respectivas licencias.
#
# Referencias:
# [1] Zhao, et al. (2023). "Learning Physically Realizable Skills for Online Packing of General 3D Shapes"
# [2] Zhao, X. (2017). "The three-dimensional container loading problem"
# [3] Parreño, F., et al. (2010). "A hybrid GRASP/VND algorithm for two-and three-dimensional bin packing"
# [4] Trimesh library: poses.compute_stable_poses, CollisionManager.
# ------------------------------------------------------------

import argparse
import os
import json
import random
import time
from collections import Counter, deque
import numpy as np
import torch
import trimesh
from trimesh.collision import CollisionManager

# === Barra de progreso ===
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, *a, **k):  # fallback silencioso
        class Dummy:
            def __init__(self, it): self.it=it
            def __iter__(self): return iter(self.it)
            def update(self, *a, **k): pass
            def set_postfix(self, *a, **k): pass
            def close(self): pass
        return Dummy(x)

class SilentProgress:
    def update(self, *a, **k): pass
    def set_postfix(self, *a, **k): pass
    def close(self): pass

# -------------------- CLI --------------------
def buffer_value(text):
    value = int(text)
    if value < 1 or value > 10:
        raise argparse.ArgumentTypeError("buffer-size debe estar entre 1 y 10")
    return value

def positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("step debe ser positivo")
    return value

def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("este valor debe ser > 0")
    return value

def non_negative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("este valor debe ser >= 0")
    return value

def parse_args():
    parser = argparse.ArgumentParser(description="Empaque secuencial con caida y buffer.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Subdirectorio dentro de ./dataset (kitchen, blockout, ...).",
    )
    parser.add_argument(
        "--buffer-size",
        type=buffer_value,
        required=True,
        help="Tamano de la ventana K.",
    )
    parser.add_argument(
        "--step",
        type=positive_float,
        required=True,
        help="Resolucion del barrido (m).",
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help="Secuencia a usar dentro de test_sequence.pt (default 0).",
    )
    parser.add_argument(
        "--restrict-rotations",
        action="store_true",
        help="Usa solo rotaciones ortogonales (desactiva poses estables de trimesh).",
    )
    parser.add_argument(
        "--tail-repack-size",
        type=non_negative_int,
        default=0,
        help="Tamano de bloque final a reordenar (0 desactiva la mejora).",
    )
    parser.add_argument(
        "--tail-repack-attempts",
        type=non_negative_int,
        default=1,
        help="Intentos deterministas para reinsertar la cola.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Semilla para reproducir comportamiento aleatorio.",
    )
    parser.add_argument(
        "--zhao-order",
        action="store_true",
        help="Reordena la secuencia usando la regla de Zhao (dimensiones/cantidad).",
    )
    parser.add_argument(
        "--max-passes",
        type=positive_int,
        default=1,
        help="Numero maximo de pasadas consecutivas reutilizando el estado.",
    )
    parser.add_argument(
        "--grasp-iterations",
        type=positive_int,
        default=1,
        help="Numero de construcciones GRASP (>=1).",
    )
    parser.add_argument(
        "--rcl-size",
        type=positive_int,
        default=3,
        help="Tamano de la lista restringida de candidatos.",
    )
    parser.add_argument(
        "--metaheuristic",
        choices=["none", "grasp", "tabu", "sa"],
        default="none",
        help="Metaheuristica aplicada sobre la secuencia inicial.",
    )
    parser.add_argument(
        "--tabu-iterations",
        type=positive_int,
        default=5,
        help="Iteraciones de Tabu Search (solo cuando metaheuristic=tabu).",
    )
    parser.add_argument(
        "--tabu-tenure",
        type=positive_int,
        default=5,
        help="Tamano de la lista tabu (en numero de swaps).",
    )
    parser.add_argument(
        "--tabu-neighborhood",
        type=positive_int,
        default=20,
        help="Maxima distancia i-j considerada en los swaps Tabu.",
    )
    parser.add_argument(
        "--tabu-candidates",
        type=positive_int,
        default=4,
        help="Numero de vecinos evaluados por iteracion Tabu.",
    )
    parser.add_argument(
        "--sa-neighborhood",
        type=positive_int,
        default=10,
        help="Distancia maxima i-j considerada en swaps para SA.",
    )
    parser.add_argument(
        "--sa-iterations",
        type=positive_int,
        default=10,
        help="Iteraciones totales para Simulated Annealing.",
    )
    parser.add_argument(
        "--sa-initial-temp",
        type=positive_float,
        default=1.0,
        help="Temperatura inicial en SA.",
    )
    parser.add_argument(
        "--sa-cooling",
        type=positive_float,
        default=0.9,
        help="Factor de enfriamiento en SA.",
    )
    return parser.parse_args()

ARGS = parse_args()
ALLOW_STABLE_POSES = not ARGS.restrict_rotations
if ARGS.random_seed is not None:
    random.seed(ARGS.random_seed)

# -------------------- Rutas --------------------
# Referencia: datasets Blockout/Kitchen/General tomados de Zhao et al. (2023) para reproducir secuencias.
# Rutas y archivos del dataset de Zhao et al. (2023).
# id2shape.pt y test_sequence.pt provienen de su distribucion;
# aqui solo se consumen para reproducibilidad de secuencias.
BASE_DIR = os.path.join("dataset", ARGS.dataset)
OBJ_DIR = os.path.join(BASE_DIR, "shape_vhacd")
SEQUENCE_PATH = os.path.join(BASE_DIR, "test_sequence.pt")
ID2NAME_PATH = os.path.join(BASE_DIR, "id2shape.pt")
RESULTS_DIR = os.path.join("resultados", ARGS.dataset)
GRASP_ACTIVE = ARGS.metaheuristic == "grasp"
OUTPUT_JSON = None  # se define al final segun metaheuristica

# ---------------- Parametros del contenedor ----------------
CONTAINER_DIMS = np.array([320, 320, 300], dtype=float) / 1000.0  # metros
STEP = float(ARGS.step)
EPS  = 1e-6   # tolerancia numerica
CONCAVE_RATIO_THRESHOLD = 0.55
CAVITY_MARGIN = 0.01
CAVITY_DEPTH_FACTOR = 0.25

# ---------------- Parametros del buffer ----------------
# Referencia: uso del concepto de buffer K inspirado en Zhao et al. (2023).
# BUFFER_SIZE: ventana de candidatos inspirada en el "buffer K" de Zhao et al. (2023).
# Adaptacion local: cola fija + seleccion codiciosa por score.
BUFFER_SIZE = max(1, min(10, int(ARGS.buffer_size)))      # rango seguro 1..10

# ---------------- Cargar datos del dataset ----------------
id2name = torch.load(ID2NAME_PATH, map_location="cpu")
sequence_data = torch.load(SEQUENCE_PATH, map_location="cpu", weights_only=False)
if isinstance(sequence_data, torch.Tensor):
    sequences = sequence_data.tolist()
else:
    sequences = sequence_data
if ARGS.sequence_index < 0 or ARGS.sequence_index >= len(sequences):
    raise IndexError(f"sequence-index {ARGS.sequence_index} fuera de rango (total {len(sequences)})")
secuencia = sequences[ARGS.sequence_index]
nombres_shapes = [id2name[int(i)] for i in secuencia]

print(f"Dataset={ARGS.dataset} | Step={STEP:.3f} m | K={BUFFER_SIZE} | Secuencia={ARGS.sequence_index}")

# ==========================================================
#                     Heightmap helpers
# ==========================================================
# Referencia: Implementacion propia — heightmap, soporte, percentil 90, cavidades, caida, score, repacks, pipeline secuencial.
# Heightmap 2D: implementación propia (discretización por STEP).
NX = max(1, int(np.floor(CONTAINER_DIMS[0] / STEP)) + 1)
NY = max(1, int(np.floor(CONTAINER_DIMS[1] / STEP)) + 1)
MAX_POSITIONS_PER_VARIANT = 40
RCL_SIZE = max(1, ARGS.rcl_size) if GRASP_ACTIVE else 1

def world_to_idx(x, y):
    ix = int(np.floor(x / STEP))
    iy = int(np.floor(y / STEP))
    ix = max(0, min(NX - 1, ix))
    iy = max(0, min(NY - 1, iy))
    return ix, iy

def rect_to_idrange(xmin, xmax, ymin, ymax):
    ix0, iy0 = world_to_idx(max(0.0, xmin + EPS), max(0.0, ymin + EPS))
    ix1, iy1 = world_to_idx(max(0.0, xmax - EPS), max(0.0, ymax - EPS))
    ix0, ix1 = min(ix0, ix1), max(ix0, ix1)
    iy0, iy1 = min(iy0, iy1), max(iy0, iy1)
    return ix0, ix1, iy0, iy1

# ==========================================================
#                    Utilidades de malla
# ==========================================================
# load_mesh_cached: carga y guarda en cache mallas 3D para evitar lecturas repetidas del disco.
# inside_container_bounds: verifica que el objeto este completamente dentro del contenedor.
# CollisionManager: módulo de trimesh usado para detectar colisiones entre objetos.
def normalize_shape_path(name):
    return os.path.normpath(name)

def load_mesh_cached(path, cache):
    if path in cache:
        return cache[path]
    m = trimesh.load(path, force='mesh')
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump().sum()
    mesh = m
    mesh_size = mesh.extents.copy()
    mesh_offset = mesh.bounds[0].copy()
    cache[path] = (mesh, mesh_size, mesh_offset)
    return cache[path]

def inside_container_bounds(mesh_world):
    mins, maxs = mesh_world.bounds
    if np.any(mins < -EPS): return False
    if np.any(maxs - CONTAINER_DIMS > EPS): return False
    return True

# ---------- Poses estables (trimesh) con fallback ----------
# Referencia: calculo de poses estables segun la API de trimesh.
# compute_stable_orientations:
# - Usa trimesh.poses.compute_stable_poses (TRIMESH) cuando esta disponible (atribución).
# - Fallback ortogonal (Rx/Ry/Rz 90°): implementación propia para cubrir poses discretas.
def compute_stable_orientations(mesh, max_poses=4):
    """
    Devuelve lista de pares (mesh_oriented, T_orient) ordenadas por probabilidad.
    T_orient es la 4x4 que orienta el mesh ORIGINAL a esa pose estable.
    Si falla, usa 4 rotaciones ortogonales como backup.
    """
    oriented = []
    try:
        if not ALLOW_STABLE_POSES:
            raise RuntimeError("stable poses disabled")
        from trimesh.poses import compute_stable_poses
        Ts, probs = compute_stable_poses(mesh, sigma=0.0, n_samples=1)
        order = np.argsort(-probs)[:max_poses]
        for i in order:
            m = mesh.copy()
            m.apply_transform(Ts[i])
            oriented.append((m, Ts[i]))
    except Exception:
        def _Rx90():
            M = np.eye(4); M[1,1]=0.0; M[1,2]=-1.0; M[2,1]=1.0; M[2,2]=0.0; return M
        def _Ry90():
            M = np.eye(4); M[0,0]=0.0; M[0,2]=1.0; M[2,0]=-1.0; M[2,2]=0.0; return M
        def _Rz90():
            M = np.eye(4); M[0,0]=0.0; M[0,1]=-1.0; M[1,0]=1.0; M[1,1]=0.0; return M
        Rs = [np.eye(4), _Rx90(), _Ry90(), _Rz90()]
        for R in Rs[:max_poses]:
            m = mesh.copy()
            m.apply_transform(R)
            oriented.append((m, R))
    return oriented

def aabb_inside_container(bounds_local, translation):
    """
    Verifica si una AABB (2x3) trasladada por 'translation' permanece dentro del contenedor.
    bounds_local corresponde al mesh orientado antes de trasladarlo.
    """
    mins = bounds_local[0] + translation
    maxs = bounds_local[1] + translation
    if np.any(mins < -EPS):
        return False
    if np.any(maxs - CONTAINER_DIMS > EPS):
        return False
    return True

def get_oriented_variants(mesh_path, mesh, cache, max_poses=4):
    """
    Devuelve (y cachea) las variantes orientadas para una malla dada.
    Cada variante guarda mesh orientado, offset, extents y matriz de orientaciA3n.
    """
    if mesh_path in cache:
        return cache[mesh_path]

    variants = []
    for oriented_mesh, T_orient in compute_stable_orientations(mesh, max_poses=max_poses):
        bounds = oriented_mesh.bounds.copy()
        variants.append({
            "mesh": oriented_mesh,
            "bounds": bounds,
            "offset": bounds[0].copy(),
            "size": oriented_mesh.extents.copy(),
            "T_orient": T_orient.copy()
        })

    if not variants:
        oriented_mesh = mesh.copy()
        bounds = oriented_mesh.bounds.copy()
        variants.append({
            "mesh": oriented_mesh,
            "bounds": bounds,
            "offset": bounds[0].copy(),
            "size": oriented_mesh.extents.copy(),
            "T_orient": np.eye(4)
        })

    cache[mesh_path] = variants
    return variants

# ==========================================================
#                Cai­da (heightmap) + reglas
# ==========================================================
# Reglas de soporte (heuri­stica propia).
MIN_SUPPORT_FRAC = 0.30  # >=30% del footprint debe estar a ~lz
SUPPORT_TOL = 0.5 * STEP # tolerancia alrededor de lz para "soporte"

def compute_lz_and_support(heightmap, xmin, xmax, ymin, ymax):
    ix0, ix1, iy0, iy1 = rect_to_idrange(xmin, xmax, ymin, ymax)
    patch = heightmap[ix0:ix1+1, iy0:iy1+1]
    if patch.size == 0:
        return 0.0, 0.0, 0.0, (ix0, ix1, iy0, iy1)
    # lz robusto: percentil 90 para evitar picos
    lz = float(np.percentile(patch, 90.0))
    # fracción de celdas "en soporte"
    support_mask = np.abs(patch - lz) <= SUPPORT_TOL
    support_frac = float(np.mean(support_mask))
    return lz, support_frac, patch, (ix0, ix1, iy0, iy1)

def candidate_score(delta_h, patch, lz, translation, size):
    # Score propio: penaliza elevación y rugosidad local + prioriza DBL
    if isinstance(patch, float):
        rough = 0.0
    else:
        rough = float(np.mean(np.abs(lz - patch)))
    lam = 0.01
    wall_gaps = [
        translation[0],
        CONTAINER_DIMS[0] - (translation[0] + size[0]),
        translation[1],
        CONTAINER_DIMS[1] - (translation[1] + size[1]),
    ]
    wall_gap = max(0.0, min(wall_gaps))
    return (
        delta_h + lam * rough,
        translation[2],
        translation[1],
        translation[0],
        wall_gap,
    )

def is_concave_container(volume_real, size):
    if size[2] <= 0:
        return False
    bounding = float(np.prod(size))
    if bounding <= 0:
        return False
    ratio = volume_real / bounding
    return ratio < CONCAVE_RATIO_THRESHOLD and size[2] > 0.05

def carve_cavity(heightmap, translation, size):
    inner_min_x = translation[0] + CAVITY_MARGIN
    inner_max_x = translation[0] + size[0] - CAVITY_MARGIN
    inner_min_y = translation[1] + CAVITY_MARGIN
    inner_max_y = translation[1] + size[1] - CAVITY_MARGIN
    if inner_min_x >= inner_max_x - EPS or inner_min_y >= inner_max_y - EPS:
        return
    ix0, ix1, iy0, iy1 = rect_to_idrange(inner_min_x, inner_max_x, inner_min_y, inner_max_y)
    cavity_height = translation[2] + min(size[2] * CAVITY_DEPTH_FACTOR, 0.04)
    heightmap[ix0:ix1+1, iy0:iy1+1] = np.minimum(heightmap[ix0:ix1+1, iy0:iy1+1], cavity_height)

def best_pos_with_drop_for_variant(variant, scene, current_height, heightmap):
    """
    Exploracion XY con caida usando lz (percentil 90) + soporte minimo.
    Trabaja con una orientacion fija y evita copiar las mallas en cada posicion.
    Devuelve un diccionario con la mejor ubicacion encontrada o None.
    """
    mesh_oriented = variant["mesh"]
    offset = variant["offset"]
    size = variant["size"]
    bounds = variant["bounds"]
    T_orient = variant["T_orient"]

    best = None
    found = 0
    stop_search = False
    limits = np.maximum(CONTAINER_DIMS - size, 0.0)

    def make_candidate(translation, score, idx_data):
        tf = np.eye(4)
        tf[:3, 3] = translation
        return {
            "translation": translation.copy(),
            "score": score,
            "variant": variant,
            "used_size": size.copy(),
            "idxs": idx_data,
            "T_world": tf @ T_orient
        }

    for y in np.arange(0, limits[1] + 1e-6, STEP):
        for x in np.arange(0, limits[0] + 1e-6, STEP):
            xmin = x; xmax = x + size[0]
            ymin = y; ymax = y + size[1]

            lz, support_frac, patch, (ix0, ix1, iy0, iy1) = compute_lz_and_support(heightmap, xmin, xmax, ymin, ymax)
            if support_frac < MIN_SUPPORT_FRAC:
                continue

            z = lz + EPS
            translation = np.array([x, y, z]) - offset
            if not aabb_inside_container(bounds, translation):
                continue

            tf = np.eye(4); tf[:3, 3] = translation
            if scene.in_collision_single(mesh_oriented, transform=tf):
                continue

            top_z = float(bounds[1][2] + translation[2])
            delta_h = max(0.0, top_z - current_height)
            sc = candidate_score(delta_h, patch, lz, translation, size)

            cand = make_candidate(translation, sc, (ix0, ix1, iy0, iy1))
            if (best is None) or (sc < best["score"]):
                best = cand
            found += 1
            if found >= MAX_POSITIONS_PER_VARIANT:
                stop_search = True
                break
        if stop_search:
            break

    if best is not None:
        base_translation = best["translation"].copy()
        base_sc = best["score"]
        base_delta_h = max(0.0, float(bounds[1][2] + base_translation[2]) - current_height)

        x0 = base_translation[0] + offset[0]
        y0 = base_translation[1] + offset[1]
        offsets = [STEP, -STEP, 2 * STEP, -2 * STEP]
        neighbors = [(x0 + dx, y0 + dy) for dx in offsets for dy in offsets if not (dx == 0 and dy == 0)]

        for xn, yn in neighbors:
            if xn < 0 or yn < 0 or xn > (CONTAINER_DIMS[0] - size[0] + 1e-9) or yn > (CONTAINER_DIMS[1] - size[1] + 1e-9):
                continue

            xmin = xn; xmax = xn + size[0]
            ymin = yn; ymax = yn + size[1]
            lz, support_frac, patch, (ix0, ix1, iy0, iy1) = compute_lz_and_support(heightmap, xmin, xmax, ymin, ymax)
            if support_frac < MIN_SUPPORT_FRAC:
                continue

            zn = lz + EPS
            translation = np.array([xn, yn, zn]) - offset
            if not aabb_inside_container(bounds, translation):
                continue

            tf = np.eye(4); tf[:3, 3] = translation
            if scene.in_collision_single(mesh_oriented, transform=tf):
                continue

            top_z_n = float(bounds[1][2] + translation[2])
            delta_h_n = max(0.0, top_z_n - current_height)
            if delta_h_n > base_delta_h + 1e-12:
                continue

            sc_n = candidate_score(delta_h_n, patch, lz, translation, size)
            if sc_n < base_sc:
                best = make_candidate(translation, sc_n, (ix0, ix1, iy0, iy1))
                base_sc = sc_n
                base_delta_h = delta_h_n

    return best

def best_feasible_position_with_drop_and_stableposes(item, scene, current_height, heightmap):
    """
    Evalua las poses orientadas cacheadas para una pieza y toma la mejor.
    Devuelve un diccionario con la seleccion o None.
    """
    best_global = None
    for variant in get_oriented_variants(item["path"], item["mesh"], stable_pose_cache, max_poses=4):
        cand = best_pos_with_drop_for_variant(variant, scene, current_height, heightmap)
        if cand is not None:
            if (best_global is None) or (cand["score"] < best_global["score"]):
                best_global = cand
    return best_global

# ==========================================================
#                    Proceso con buffer
# ==========================================================
# Referencia: pipeline de empaque secuencial/online inspirado en Zhao et al. (2023).
def run_packing(sequence_ids, mesh_cache, stable_pose_cache, show_progress=True, initial_state=None):
    if initial_state is not None:
        placed = list(initial_state.get("placements", []))
        scene = initial_state["scene"]
        heightmap = initial_state["heightmap"]
        volumen_usado = float(initial_state.get("volumen_usado", 0.0))
        current_height = float(initial_state.get("current_height", 0.0))
        volumen_total = float(initial_state.get("volumen_total", np.prod(CONTAINER_DIMS)))
    else:
        placed = []
        # Referencia: uso de CollisionManager segun la API de trimesh.
        scene = CollisionManager()
        heightmap = np.zeros((NX, NY), dtype=np.float64)
        volumen_total = np.prod(CONTAINER_DIMS)
        volumen_usado = 0.0
        current_height = 0.0

    buffer = []
    idx = 0
    no_fit_rounds = 0

    total_items = len(sequence_ids)
    pbar = tqdm(total=total_items, desc="Empaquetando objetos", ncols=90) if show_progress else SilentProgress()

    start_time = time.perf_counter()
    while (idx < total_items) or buffer:
        while idx < total_items and len(buffer) < BUFFER_SIZE:
            name = sequence_ids[idx]
            path_obj = os.path.join(OBJ_DIR, normalize_shape_path(name))
            mesh, mesh_size, mesh_offset = load_mesh_cached(path_obj, mesh_cache)
            buffer.append({
                "name": name,
                "path": path_obj,
                "mesh": mesh,
                "mesh_size": mesh_size,
                "mesh_offset": mesh_offset
            })
            idx += 1

        if not buffer:
            break

        # Referencia: estructura GRASP (greedy + RCL + aleatoriedad) adaptada de Parreño et al. (2010).
        candidate_entries = []
        for j, item in enumerate(buffer):
            cand = best_feasible_position_with_drop_and_stableposes(item, scene, current_height, heightmap)
            if cand is not None:
                candidate_entries.append((cand["score"], random.random(), j, cand))

        if not candidate_entries:
            no_fit_rounds += 1
            if idx < total_items:
                buffer.pop(0)
                continue
            elif no_fit_rounds > 2:
                break
            else:
                buffer = buffer[1:] + buffer[:1]
                continue
        if GRASP_ACTIVE:
            candidate_entries.sort(key=lambda x: (x[0], x[1]))
            rcl = candidate_entries[:RCL_SIZE]
            chosen = random.choice(rcl)
            _, _, j, best_data = chosen
        else:
            candidate_entries.sort(key=lambda x: x[0])
            _, _, j, best_data = candidate_entries[0]
        no_fit_rounds = 0
        item = buffer.pop(j)

        unique_key = f"{item['name']}#{len(placed)}"
        variant = best_data["variant"]
        translation = best_data["translation"]
        used_size = best_data["used_size"]
        idr = best_data["idxs"]
        tf_local = np.eye(4)
        tf_local[:3, 3] = translation
        scene.add_object(unique_key, variant["mesh"], transform=tf_local)

        placed.append({
            "id": item["name"],
            "position_m": list(np.round(translation, 4)),
            "size_m": list(np.round(used_size, 4)),
            "transform_matrix": np.round(best_data["T_world"], 6).tolist(),
            "T_world": np.round(best_data["T_world"], 6).tolist()
        })

        volumen_usado += item["mesh"].volume
        bounds = variant["bounds"]
        top_z = float(bounds[1][2] + translation[2])
        current_height = max(current_height, top_z)

        ix0, ix1, iy0, iy1 = idr
        heightmap[ix0:ix1+1, iy0:iy1+1] = np.maximum(heightmap[ix0:ix1+1, iy0:iy1+1], top_z)
        if is_concave_container(item["mesh"].volume, used_size):
            carve_cavity(heightmap, translation, used_size)

        pct = (volumen_usado / volumen_total * 100.0) if volumen_total > 0 else 0.0
        pbar.update(1)
        pbar.set_postfix(vol=f"{pct:5.2f}%", h=f"{current_height:.3f}m")

    pbar.close()
    remaining_sequence = [item["name"] for item in buffer] + sequence_ids[idx:]
    elapsed = time.perf_counter() - start_time
    fill_percent = (volumen_usado / volumen_total * 100.0) if volumen_total > 0 else 0.0
    return {
        "placements": placed,
        "volume_total": volumen_total,
        "volume_used": volumen_usado,
        "fill_percent": fill_percent,
        "placed_count": len(placed),
        "elapsed": elapsed,
        "order": list(sequence_ids),
        "remaining": remaining_sequence,
        "state": {
            "placements": placed,
            "scene": scene,
            "heightmap": heightmap,
            "volumen_usado": volumen_usado,
            "current_height": current_height,
            "volumen_total": volumen_total,
        },
    }

def better_result(candidate, current):
    if current is None:
        return True
    if candidate["placed_count"] > current["placed_count"]:
        return True
    if candidate["placed_count"] < current["placed_count"]:
        return False
    if candidate["fill_percent"] > current["fill_percent"] + 1e-6:
        return True
    if candidate["fill_percent"] + 1e-6 < current["fill_percent"]:
        return False
    return candidate["elapsed"] < current["elapsed"]


mesh_cache = {}
shape_stats = {}
if ARGS.zhao_order:
    # Referencia: regla de ordenamiento secuencial basada en Zhao (2017).
    type_counts = Counter(nombres_shapes)
    shape_stats = {}
    for name in type_counts:
        path_obj = os.path.join(OBJ_DIR, normalize_shape_path(name))
        mesh, mesh_size, mesh_offset = load_mesh_cached(path_obj, mesh_cache)
        extents = mesh_size
        shape_stats[name] = {
            "min_dim": float(np.min(extents)),
            "max_dim": float(np.max(extents)),
            "count": type_counts[name],
        }

    def zhao_key(name):
        stats = shape_stats.get(name)
        if not stats:
            return (0.0, 0.0, 0.0)
        return (-stats["min_dim"], -stats["count"], -stats["max_dim"])

    nombres_shapes = sorted(nombres_shapes, key=zhao_key)
else:
    type_counts = Counter(nombres_shapes)

stable_pose_cache = {}

def run_with_passes(order, show_first_pass=True):
    remaining_sequence = list(order)
    state = None
    best_result_local = None
    for pass_idx in range(max(1, ARGS.max_passes)):
        if not remaining_sequence:
            break
        show = show_first_pass and (pass_idx == 0)
        result = run_packing(
            remaining_sequence,
            mesh_cache,
            stable_pose_cache,
            show_progress=show,
            initial_state=state,
        )
        best_result_local = result
        state = result["state"]
        remaining_sequence = result["remaining"]
        if not remaining_sequence:
            break
    return best_result_local

def solve_order(order, show_first_pass=True):
    best_result_local = None
    iterations = max(1, ARGS.grasp_iterations) if GRASP_ACTIVE else 1
    # Referencia: estructura GRASP (greedy + RCL + aleatoriedad) adaptada de Parreño et al. (2010).
    for iter_idx in range(iterations):
        show = show_first_pass and (iter_idx == 0)
        candidate = run_with_passes(order, show_first_pass=show)
        if GRASP_ACTIVE:
            print(
                f"GRASP iter {iter_idx + 1}/{iterations}: "
                f"{candidate['placed_count']} piezas, {candidate['fill_percent']:.2f}%"
            )
        if better_result(candidate, best_result_local):
            best_result_local = candidate
    return best_result_local

def apply_tail_repack(base_result, base_order):
    if (
        ARGS.max_passes != 1
        or ARGS.tail_repack_size <= 0
        or len(base_order) <= ARGS.tail_repack_size
    ):
        return base_result, list(base_order)
    tail_size = min(ARGS.tail_repack_size, len(base_order) - 1)
    max_attempts = min(len(base_order) - tail_size + 1, max(1, ARGS.tail_repack_attempts))
    best_res = base_result
    best_ord = list(base_order)
    for attempt in range(max_attempts):
        start_tail = max(0, len(base_order) - tail_size - attempt)
        block = base_order[start_tail:start_tail + tail_size]
        remaining = base_order[:start_tail] + base_order[start_tail + tail_size:]
        if not block or not remaining:
            continue
        new_order = block + remaining
        candidate = solve_order(new_order, show_first_pass=False)
        if better_result(candidate, best_res):
            print(
                f"Tail repack intento {attempt + 1}: {candidate['placed_count']} piezas, "
                f"{candidate['fill_percent']:.2f}% lleno"
            )
            best_res = candidate
            best_ord = new_order
    return best_res, best_ord

def run_tabu_search(initial_order):
    if len(initial_order) < 2:
        return solve_order(initial_order), list(initial_order)
    best_order = list(initial_order)
    best_res = solve_order(best_order)
    current_order = list(best_order)
    tabu_queue = deque(maxlen=ARGS.tabu_tenure)
    for iteration in range(ARGS.tabu_iterations):
        neighbors = []
        for _ in range(ARGS.tabu_candidates):
            if len(current_order) < 2:
                break
            i = random.randint(0, len(current_order) - 2)
            j_max = min(len(current_order) - 1, i + ARGS.tabu_neighborhood)
            j = random.randint(i + 1, j_max)
            move = (min(i, j), max(i, j))
            if move in tabu_queue:
                continue
            neighbor = current_order[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append((move, neighbor))
        if not neighbors:
            break
        best_neighbor_move = None
        best_neighbor_order = None
        best_neighbor_res = None
        for move, neighbor in neighbors:
            candidate = solve_order(neighbor, show_first_pass=False)
            if best_neighbor_res is None or better_result(candidate, best_neighbor_res):
                best_neighbor_res = candidate
                best_neighbor_move = move
                best_neighbor_order = neighbor
        if best_neighbor_order is None:
            break
        tabu_queue.append(best_neighbor_move)
        current_order = best_neighbor_order
        if better_result(best_neighbor_res, best_res):
            print(
                f"Tabu mejora a {best_neighbor_res['placed_count']} piezas, "
                f"{best_neighbor_res['fill_percent']:.2f}%"
            )
            best_res = best_neighbor_res
            best_order = best_neighbor_order
    return best_res, best_order

def run_simulated_annealing(initial_order):
    if len(initial_order) < 2:
        return solve_order(initial_order), list(initial_order)
    current_order = list(initial_order)
    current_res = solve_order(current_order)
    best_order = list(current_order)
    best_res = current_res
    temperature = ARGS.sa_initial_temp
    iterations = max(1, ARGS.sa_iterations)

    def energy(res):
        if res is None:
            return float("inf")
        return -(res["placed_count"] * 1000.0 + res["fill_percent"])

    for iter_idx in range(iterations):
        if len(current_order) < 2:
            break
        i = random.randint(0, len(current_order) - 2)
        j = random.randint(i + 1, min(len(current_order) - 1, i + ARGS.sa_neighborhood))
        neighbor = current_order[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        candidate = run_with_passes(neighbor, show_first_pass=False)
        delta = energy(candidate) - energy(current_res)
        if delta < 0 or random.random() < np.exp(-delta / max(temperature, 1e-6)):
            current_order = neighbor
            current_res = candidate
            print(
                f"SA iter {iter_idx + 1}/{iterations}: "
                f"{candidate['placed_count']} piezas, {candidate['fill_percent']:.2f}%"
            )
            if better_result(candidate, best_res):
                best_res = candidate
                best_order = neighbor
        temperature *= ARGS.sa_cooling
        if temperature < 1e-4:
            break
    return best_res, best_order

if ARGS.metaheuristic == "grasp":
    best_result = solve_order(nombres_shapes)
    best_order = list(nombres_shapes)
    best_result, best_order = apply_tail_repack(best_result, best_order)
elif ARGS.metaheuristic == "tabu":
    best_result, best_order = run_tabu_search(nombres_shapes)
elif ARGS.metaheuristic == "sa":
    best_result, best_order = run_simulated_annealing(nombres_shapes)
else:
    best_result = solve_order(nombres_shapes)
    best_order = list(nombres_shapes)
    best_result, best_order = apply_tail_repack(best_result, best_order)

placed = best_result["placements"]
volumen_total = best_result["volume_total"]
volumen_usado = best_result["volume_used"]
fill_percent = best_result["fill_percent"]
elapsed = best_result["elapsed"]

method_tag = ARGS.metaheuristic if ARGS.metaheuristic != "none" else ARGS.dataset
step_tag = f"s{int(ARGS.step * 1000):04d}"
OUTPUT_JSON = os.path.join(RESULTS_DIR, f"solucion_{method_tag}_k{BUFFER_SIZE}_{step_tag}.json")
meta_dir = os.path.join(RESULTS_DIR, "meta")
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(placed, f, indent=4)
print(f"\nGuardado {len(placed)} objetos en {OUTPUT_JSON}")

meta_path = os.path.join(meta_dir, f"meta_{method_tag}_k{BUFFER_SIZE}_{step_tag}.json")
metadata = {
    "dataset": ARGS.dataset,
    "buffer_size": BUFFER_SIZE,
    "step": STEP,
    "sequence_index": ARGS.sequence_index,
    "metaheuristic": ARGS.metaheuristic,
    "grasp_iterations": ARGS.grasp_iterations,
    "rcl_size": ARGS.rcl_size,
    "max_passes": ARGS.max_passes,
    "tail_repack_size": ARGS.tail_repack_size,
    "tail_repack_attempts": ARGS.tail_repack_attempts,
    "random_seed": ARGS.random_seed,
    "placed": len(placed),
    "fill_percent": fill_percent,
    "volume_total": volumen_total,
    "volume_used": volumen_usado,
    "elapsed": elapsed,
    "output_json": os.path.relpath(OUTPUT_JSON, start=RESULTS_DIR),
}
if ARGS.metaheuristic == "tabu":
    metadata["tabu_params"] = {
        "iterations": ARGS.tabu_iterations,
        "tenure": ARGS.tabu_tenure,
        "neighborhood": ARGS.tabu_neighborhood,
        "candidates": ARGS.tabu_candidates,
    }
if ARGS.metaheuristic == "sa":
    metadata["sa_params"] = {
        "iterations": ARGS.sa_iterations,
        "initial_temp": ARGS.sa_initial_temp,
        "cooling": ARGS.sa_cooling,
        "neighborhood": ARGS.sa_neighborhood,
    }
with open(meta_path, "w", encoding="utf-8") as meta_file:
    json.dump(metadata, meta_file, indent=4)
print(f"Metadata en {meta_path}")

volumen_usado_chk = 0.0
for p in placed:
    m = trimesh.load(os.path.join(OBJ_DIR, normalize_shape_path(p["id"])), force='mesh')
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump().sum()
    volumen_usado_chk += m.volume

print(f"\n Objetos colocados: {len(placed)}")
print(f"Volumen contenedor: {volumen_total:.6f} m³")
print(f"Volumen usado:     {volumen_usado_chk:.6f} m³")
print(f"Porcentaje lleno:  {(volumen_usado_chk / volumen_total) * 100:.2f}%")
