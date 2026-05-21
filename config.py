
# fisica
GRAVITY = 9.8

#integrador Leapfrog
DT        = 0.02
MAX_STEPS = 20_000

# condiciones iniciales
V0_MIN, V0_MAX       = 5.0, 30.0
THETA_MIN, THETA_MAX = 15.0, 75.0
K_MIN, K_MAX         = 0.001, 0.05

# dataset
N_TRAJECTORIES = 500
N_SMALL        = 100
RANDOM_SEED    = 42

# modelo ML
TEST_SIZE         = 0.2
N_ESTIMATORS      = 30
MAX_DEPTH         = 10
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF  = 2
CV_FOLDS          = 3

# analisis de sensibilidad
DT_VALUES = [0.01, 0.02, 0.05, 0.1]
K_VALUES  = [0.001, 0.01, 0.02, 0.05]

# rutas para las salidas
import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR    = os.path.join(BASE_DIR, "data")

DATASET_PATH = os.path.join(DATA_DIR, "dataset_proyectil.csv")
FIGURE_PATH  = os.path.join(OUTPUT_DIR, "resultados_proyectil_ML.png")
MODEL_PATH   = os.path.join(OUTPUT_DIR, "random_forest.joblib")