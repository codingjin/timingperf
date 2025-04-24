import statistics
import torch
import cudnn
import argparse

"""
N, C, H, W = int(arguments['N']), int(arguments['C']), int(arguments['H']), int(arguments['W'])
K, R, S = int(arguments['K']), int(arguments['R']), int(arguments['S'])
P, Q = H-R+1, W-S+1
"""

N, C, H, W = 8, 3, 2048, 2048
K, R, S = 64, 7, 7
P, Q = H-R+1, W-S+1

print(f"N={N} C={C} H={H} W={W}")
print(f"K={K} R={R} S={S} P={P} Q={Q}")

torch.manual_seed(149)

handle = cudnn.create_handle()
graph = cudnn.pygraph(
    handle=handle,
    name="cudnn_conv_graph",
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
)

X = graph.tensor(
    name="X",
    dim=[N, C, H, W],
    stride=[C*H*W, H*W, W, 1],
    data_type=cudnn.data_type.HALF,
)

W = graph.tensor(name="W", dim=[K, C, R, S], stride=[C*R*S, R*S, S, 1])

Y = graph.conv_fprop(
    X,
    W,
    padding=[0, 0],
    stride=[1, 1],
    dilation=[1, 1],
    compute_data_type=cudnn.data_type.FLOAT,
)
Y.set_output(True)
graph.build([cudnn.heur_mode.A])

X_gpu = torch.randn(
    8, 3, 2048, 2048, requires_grad=False, device="cuda", dtype=torch.float16
)
W_gpu = torch.randn(
    64, 3, 7, 7, requires_grad=False, device="cuda", dtype=torch.float16
)
Y_gpu = torch.zeros(
    8, 64, 2042, 2042, requires_grad=False, device="cuda", dtype=torch.float16
)
workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

# warp-up 3 rounds
for i in range(3):
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_gpu}, workspace, handle=handle)


# 9 rounds
time_array = []
for i in range(9):
    start_torch = torch.cuda.Event(enable_timing=True)
    end_torch = torch.cuda.Event(enable_timing=True)
    start_torch.record()
    graph.execute({X: X_gpu, W: W_gpu, Y: Y_gpu}, workspace, handle=handle)
    end_torch.record()
    torch.cuda.synchronize()
    torch_time = start_torch.elapsed_time(end_torch)
    time_array.append(torch_time)

median_time = statistics.median(time_array)
print(f"Pytorch conv Performance is {2*0.000001*N*K*P*Q*C*R*S/median_time:.0f} GFLOPs")

print("Complete!")
print()