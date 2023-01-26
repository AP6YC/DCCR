# using PyCall
using Pkg

# Point to the correct Python environment
if !haskey(ENV, "PYTHON")
    PYTHON_ENV = raw"C:\Users\Sasha\Anaconda3\envs\l2mmetrics\python.exe"
    if isfile(PYTHON_ENV)
        ENV["PYTHON"] = PYTHON_ENV
    else
        ENV["PYTHON"] = ""
    end
end

Pkg.build("PyCall")

# Load PyCall and Conda after setting the default environment
using PyCall
