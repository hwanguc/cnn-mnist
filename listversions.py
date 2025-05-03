"""
listversions.py
---------------------------------
Generate a requirements-style list of the exact versions that are
**already installed** in your current Python environment.

Usage:
    python listversions.py  > requirements.txt
"""

import importlib
import pkg_resources

PKGS = [
    # --- core ---
    "streamlit",
    "streamlit_drawable_canvas",

    # --- ML / numeric ---
    "torch",
    "torchvision",
    "torchmetrics",
    "torchsummary",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "opencv_python_headless",
    "h5py",
    "pillow",

    # --- plotting ---
    "altair",

    # --- database ---
    "psycopg2_binary",

    # --- utility ---
    "python_dotenv",
]

lines = []
for name in PKGS:
    # pkg_resources needs the PyPI-style name with dashes, not underscores
    pypi_name = name.replace("_", "-")
    try:
        # Try import first (handles cases where dist name ≠ import name)
        mod = importlib.import_module(name.replace("-", "_"))
        version = getattr(mod, "__version__", None)
        if not version:
            version = pkg_resources.get_distribution(pypi_name).version
        lines.append(f"{pypi_name}=={version}")
    except Exception:
        # not installed in this environment; skip it
        pass

# alphabetical for readability
for line in sorted(lines, key=str.lower):
    print(line)