from pathlib import Path

# # Point to the top of the project relative to this script
def projectdir(*args):
    return str(Path.cwd().joinpath("..", "..", "..", *args).resolve())

def print_allocated_memory():
   print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))
