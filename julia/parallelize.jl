# Start several processes
using Distributed
addprocs(3, exeflags="--project=.")
