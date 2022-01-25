# Start several processes
using Distributed
addprocs(26, exeflags="--project=.")
