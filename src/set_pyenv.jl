"""
    set_pyenv.jl

# Description
Sets the Python environment variable in an idempotent manner.
"""

# Point to the correct Python environment
if !haskey(ENV, "PYTHON")
    # PYTHON_ENV = raw"C:\Users\Sasha\Anaconda3\envs\l2mmetrics\python.exe"
    # PYTHON_ENV = raw"C:\Users\sap62\Anaconda3\envs\l2m\python.exe"
    # PYTHON_ENV = raw"C:\Users\sap62\Anaconda3\envs\l2m\python.exe"
    # PYTHON_ENV = raw"/home/sap625/miniconda3/envs/l2m/bin/python"
    PYTHON_ENV = raw"/home/sap625/envs/l2m/bin/python"
    if isfile(PYTHON_ENV)
        ENV["PYTHON"] = PYTHON_ENV
    else
        ENV["PYTHON"] = ""
    end
end
