# Procedimiento para usar el programa 
 * Crear un enviroment Python 3.11.6
 * .\.venv\Scripts\activate
 * pip install -r requirements.txt

# Ejecución principal (empaque secuencial mejorado)
Usa `upgraded_secuencial.py`, que soporta metaheurísticas (`--metaheuristic grasp/tabu/sa/none`), buffer K y distintos datasets (general, kitchen, blockout).

Ejemplo básico (grasp, k=3, step=0.02, dataset general):
```
python upgraded_secuencial.py --dataset general --buffer-size 3 --step 0.02 --sequence-index 3 --restrict-rotations --zhao-order --metaheuristic grasp --grasp-iterations 10 --rcl-size 8 --max-passes 2 --random-seed 42
```
Salida: `resultados/<dataset>/solucion_<meta>_k<k>_s####.json` y metadata en `resultados/<dataset>/meta/`.

# Scripts de experimentos
- `run_experiments_general.py`, `run_experiments_kitchen.py`, `run_experiments_blockout.py`: barren buffers [1,3,5,10] y steps [0.01,0.02,0.03] con configuración fija por dataset. Ejecuta desde la raíz:
```
python run_experiments_general.py
python run_experiments_kitchen.py
python run_experiments_blockout.py
```

# Medición rápida de tiempo (una pasada por combinación)
- `Single_run_timing.py`: corre todas las combinaciones (3 datasets x 4 buffers x 3 steps) con 1 iteración de GRASP y registra tiempos en `resultados/tiempos_single.csv`.
```
python Single_run_timing.py
```

# Graficar tiempos
- `plot_tiempos_single.py`: lee `resultados/tiempos_single.csv` y genera `resultados/tiempos_single.png` con curvas de tiempo por dataset/step.
```
python plot_tiempos_single.py
```


