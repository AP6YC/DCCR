# Start several processes
using Distributed
addprocs(22, exeflags="--project=.")
