import sys
import statistics
import torch

arguments = sys.argv[1:]

assert len(arguments) == 4

batch_size, M, N, K = int(arguments[0]), int(arguments[1]), int(arguments[2]), int(arguments[3])

torch.manual_seed(149)

A = torch.randn((batch_size, M, K), device='cuda', dtype=torch.float16)
B = torch.randn((batch_size, K, N), device='cuda', dtype=torch.float16)
C = torch.zeros((batch_size, M, N), device='cuda', dtype=torch.float16)


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
print(f"batch_size={batch_size} M={M} N={N} K={K}")
print(f"Pytorch Matrix-multiplication Performance is {2*batch_size*M*N*K/median_time/1000000:.0f} GFLOPs")
