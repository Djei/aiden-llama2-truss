import truss
from pathlib import Path

tr = truss.load("../aiden-llama2-truss")
command = tr.docker_build_setup(build_dir=Path("../aiden-llama2-truss/build/"))
print(command)