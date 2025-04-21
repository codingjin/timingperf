import sys
import statistics
import torch

arguments = sys.argv[1:]

assert len(arguments) == 3

M, N, K = int(arguments[0]), int(arguments[1]), int(arguments[2])

torch.manual_seed(149)

A = torch.randn((M, K), device='cuda', dtype=torch.float16)
B = torch.randn((K, N), device='cuda', dtype=torch.float16)
C = torch.zeros((M, N), device='cuda', dtype=torch.float16)


# Warm-up
for i in range(3):
    C = torch.matmul(A, B)
time_array = []
for i in range(9):
    start_torch = torch.cuda.Event(enable_timing=True)
    end_torch = torch.cuda.Event(enable_timing=True)
    start_torch.record()
    C = torch.matmul(A, B)
    end_torch.record()
    torch.cuda.synchronize()
    torch_time = start_torch.elapsed_time(end_torch)
    time_array.append(torch_time)

#print(time_array)
#print(sorted(time_array))
#print(statistics.median(time_array))
median_time = statistics.median(time_array)
print(f"M={M} N={N} K={K}")
print(f"Pytorch Matrix-multiplication Performance is {2*M*N*K/median_time/1000000:.0f} GFLOPs")
