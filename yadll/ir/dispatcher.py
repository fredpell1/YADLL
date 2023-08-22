import os
import importlib
backend = os.getenv("backend", "numpy")
imports = {
    "numpy": "yadll.backends.numpy_backend"
}
if backend not in imports.keys():
    print(f"backend {backend} currently not supported. Defaulting to numpy")
    backend = "numpy"
module_name = imports[backend]
module = importlib.import_module(module_name)
# Import everything from the module
for attr_name in dir(module):
    if not attr_name.startswith("_"):  # Skip private attributes (those starting with an underscore)
        globals()[attr_name] = getattr(module, attr_name)
