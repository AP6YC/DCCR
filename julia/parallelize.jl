# Start several processes
using Distributed
addprocs(15, exeflags="--project=.")
