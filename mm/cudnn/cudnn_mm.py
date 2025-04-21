import sys
import statistics
import torch
import cudnn

arguments = sys.argv[1:]

assert len(arguments) == 3

batch_size, M, N, K = int(1), int(arguments[0]), int(arguments[1]), int(arguments[2])

torch.manual_seed(149)

A = torch.randn((batch_size, M, K), device='cuda', dtype=torch.float16)
B = torch.randn((batch_size, K, N), device='cuda', dtype=torch.float16)
C = torch.zeros((batch_size, M, N), device='cuda', dtype=torch.float16)

# Create cudnn graph and tensors
graph = cudnn.pygraph()
A_cudnn_tensor = graph.tensor_like(A)
B_cudnn_tensor = graph.tensor_like(B)

C_cudnn_tensor = graph.matmul(
    name='matmul',
    A=A_cudnn_tensor,
    B=B_cudnn_tensor,
    compute_data_type=cudnn.data_type.FLOAT,
)
C_cudnn_tensor.set_name("C").set_output(True).set_data_type(torch.float16)

# Build the graph
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
graph.check_support()
graph.build_plans()

# Execute the code
variant_pack = {
    A_cudnn_tensor: A,
    B_cudnn_tensor: B,
    C_cudnn_tensor: C,
}

workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

# Warm-up
for i in range(3):
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

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

median_time = statistics.median(time_array)
print(f"batch_size={batch_size} M={M} N={N} K={K}")
print(f"CUDNN Matrix-multiplication Performance is {2*batch_size*M*N*K/median_time/1000000:.0f} GFLOPs")



