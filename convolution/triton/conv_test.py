import sys
import statistics
import torch
import triton
import triton.language as tl

import argparse

def parse_key_value_pair(pair):
    try:
        key, value = pair.split('=')
        return key, value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Argument '{pair} must be in format KEY=VALUE")

parser = argparse.ArgumentParser(description='Arguments for triton matrix-multiplication')
parser.add_argument('params', nargs='+', type=parse_key_value_pair, help='Parameters in format KEY=VALUE')
args = parser.parse_args()
arguments = dict(args.params)

required_args = ['N', 'C', 'H', 'W', 'K', 'R', 'S']
optional_args = {
    'stride': 1,
    'padding': 0,
    'dilation': 1
}
#required_args = ['N', 'C', 'H', 'W', 'K', 'R', 'S', 'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M', 'num_warps']
missing = [arg for arg in required_args if arg not in arguments]
if missing:
    parser.error(f"The following required arguments are missing: {', '.join(missing)}")

# Apply default values for optional args if not provided
for key, default_value in optional_args.items():
    if key not in arguments:
        arguments[key] = str(default_value)

# All integer arguments
for key in arguments:
    if arguments[key].isdigit():
        arguments[key] = int(arguments[key])

#print(arguments)
assert len(arguments) == 10

N, C, H, W = arguments['N'], arguments['C'], arguments['H'], arguments['W']
K, R, S = arguments['K'], arguments['R'], arguments['S']
stride, padding, dilation = arguments['stride'], arguments['padding'], arguments['dilation']
P, Q = (H + 2*padding - dilation*(R - 1) - 1)//stride + 1, (W + 2*padding - dilation*(S - 1) - 1)//stride + 1

BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps = 128, 128, 32, 8, 4
print(f"N={N} C={C} H={H} W={W}")
print(f"K={K} R={R} S={S}")
print(f"P={P} Q={Q}")
print(f"stride={stride} padding={padding} dilation={dilation}")
print(f"BLOCK_SIZE_M={BLOCK_SIZE_M} BLOCK_SIZE_N={BLOCK_SIZE_N} BLOCK_SIZE_K={BLOCK_SIZE_K} GROUP_SIZE_M={GROUP_SIZE_M} num_warps={num_warps}")


@triton.autotune(
    configs = [triton.Config({'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_N': BLOCK_SIZE_N, 'BLOCK_SIZE_K': BLOCK_SIZE_K, 'GROUP_SIZE_M': GROUP_SIZE_M}, 
                              num_stages=2, num_warps=num_warps)],
    key=['N', 'C', 'H', 'W', 'K', 'R', 'S', 'P', 'Q', 'stride', 'padding', 'dilation'],
)
@triton.jit
def conv_kernel(input_ptr, kernel_ptr, output_ptr, 
                N, C, H, W, K, R, S, P, Q, stride, padding, dilation, 
                GEMM_M, GEMM_N, GEMM_K, 
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m, num_pid_n = tl.cdiv(GEMM_M, BLOCK_SIZE_M), tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M*num_pid_n
    group_id, group_offset = pid // num_pid_in_group, pid % num_pid_in_group
    group_size_m = min(GROUP_SIZE_M, num_pid_m-group_id*GROUP_SIZE_M)
    pid_m, pid_n = group_id*GROUP_SIZE_M+group_offset%group_size_m, group_offset//group_size_m

    gemm_m, gemm_n = (pid_m*BLOCK_SIZE_M+tl.arange(0, BLOCK_SIZE_M))%GEMM_M, (pid_n*BLOCK_SIZE_N+tl.arange(0, BLOCK_SIZE_N))%GEMM_N

    n, npq_res = gemm_m//(P*Q), gemm_m%(P*Q)
    p, q = npq_res//Q, npq_res%Q
    k = gemm_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    gemm_k_bound = tl.cdiv(GEMM_K, BLOCK_SIZE_K)
    for ik in range(0, gemm_k_bound):
        gemm_k = ik*BLOCK_SIZE_K+tl.arange(0, BLOCK_SIZE_K)
        c, crs_res = gemm_k//(R*S), gemm_k%(R*S)
        r, s = crs_res//S, crs_res%S

        h = p[:, None]*stride + r[None, :]*dilation - padding
        w = q[:, None]*stride + s[None, :]*dilation - padding

        input_offsets = n[:, None]*C*H*W + c[None, :]*H*W + h*W + w
        kernel_offsets = k[None, :]*C*R*S + c[:, None]*R*S + r[:, None]*S + s[:, None]

        input_ptrs, kernel_ptrs = input_ptr+input_offsets, kernel_ptr+kernel_offsets
        input_mask, kernel_mask = (h >= 0) & (h < H) & (w >= 0) & (w < W), (r[:, None] < R) & (s[:, None] < S) & (c[:, None] < C)
        input, kernel = tl.load(input_ptrs, mask=input_mask, other=0.0), tl.load(kernel_ptrs, mask=kernel_mask, other=0.0)
        accumulator = tl.dot(input, kernel, accumulator)

    output_offsets = n[:, None]*K*P*Q + k[None, :]*P*Q + p[:, None]*Q + q[:, None]
    output_mask = (n[:, None]<N) & (k[None, :]<K) & (p[:, None]<P) & (q[:, None]<Q)
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=output_mask)


torch.manual_seed(149)
# NCHW
input = torch.randn(size=(N, C, H, W), dtype=torch.float16, device='cuda')
# KCRS
kernel = torch.randn(size=(K, C, R, S), dtype=torch.float16, device='cuda')
# NKPQ
output = torch.zeros(size=(N, K, P, Q), dtype=torch.float16, device='cuda')

GEMM_M, GEMM_N, GEMM_K = N*P*Q, K, C*R*S
grid = lambda meta: (triton.cdiv(GEMM_M, meta['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, meta['BLOCK_SIZE_N']), )
conv_kernel[grid](input, kernel, output, 
            N, C, H, W, K, R, S, P, Q, stride, padding, dilation, 
            GEMM_M, GEMM_N, GEMM_K)



output_torch = torch.nn.functional.conv2d(input, kernel, stride=stride, padding=padding, dilation=dilation)

print("Max diff: ", (output - output_torch).abs().max())

"""
# warm-up 3 rounds 
for i in range(3):
    conv_kernel[grid](input, kernel, output, 
            N, C, H, W, K, R, S, P, Q, 
            GEMM_M, GEMM_N, GEMM_K)
# 9 rounds
time_array = []
for i in range(9):
    start_torch = torch.cuda.Event(enable_timing=True)
    end_torch = torch.cuda.Event(enable_timing=True)
    start_torch.record()
    conv_kernel[grid](input, kernel, output, 
            N, C, H, W, K, R, S, P, Q, 
            GEMM_M, GEMM_N, GEMM_K)
    end_torch.record()
    torch.cuda.synchronize()
    torch_time = start_torch.elapsed_time(end_torch)
    time_array.append(torch_time)

median_time = statistics.median(time_array)
print(f"Triton conv2d Performance is {2*0.000001*N*K*P*Q*C*R*S/median_time:.0f} GFLOPs\n")
"""