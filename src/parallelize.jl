# Start several processes
using Distributed
addprocs(24, exeflags="--project=.")
