### instalacion y ejecucion

#### requisitos

- Python 3.8 o superior
- pip

#### clonar el repositorio

```bash
git clone https://github.com/arcibyte/projectile_leapfrog.git
cd projectile_leapfrog
```

#### instalar dependencias

```bash
pip install -r requirements.txt
```
las dependencias del proyecto son:
numpy
pandas
scikit-learn
joblib
matplotlib

#### ejecutar el pipeline completo

```bash
python proyectile.py
```

esto ejecuta en orden: generación del dataset → entrenamiento del modelo → 
evaluación por rollout → análisis de sensibilidad → generación de figuras. 
Todos los resultados se guardan en `outputs/`.
