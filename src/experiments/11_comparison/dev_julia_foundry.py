import julia

runtime="/home/sap625/julia"

from pathlib import Path

# # Point to the top of the project relative to this script
def projectdir(*args):
    return str(Path.cwd().joinpath("..", "..", "..", *args).resolve())

julia.install(julia=runtime)

print("--- Setting up Julia runtime ---")

# Setup the runtime
jl_run = julia.Julia(
    runtime=runtime,
    compiled_modules=False,
    # {"project": projectdir},
)

print("--- Activating environment ---")

from julia import Pkg
Pkg.activate(projectdir())

# Point to the actual global namespace
from julia import Main
jl = Main

jl.x = [1,2,3]
print(jl.eval("sin.(x)"))

# jl.project_dir = projectdir()
# jl.eval("using Pkg; Pkg.activate(project_dir)")

