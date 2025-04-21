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

required_args = ['batch_size', 'M', 'N', 'K', 'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M', 'num_warps']
missing = [arg for arg in required_args if arg not in arguments]
if missing:
    parser.error(f"The following required arguments are missing: {', '.join(missing)}")

for key in arguments:
    if arguments[key].isdigit():
        arguments[key] = int(arguments[key])

#print(arguments)
assert len(arguments) == 9

batch_size, M, N, K = int(arguments['batch_size']), int(arguments['M']), int(arguments['N']), int(arguments['K'])
bm_size, bn_size, bk_size, gp_size = int(arguments['BLOCK_SIZE_M']), int(arguments['BLOCK_SIZE_N']), int(arguments['BLOCK_SIZE_K']), int(arguments['GROUP_SIZE_M'])
num_warps = arguments['num_warps']

torch.manual_seed(149)
A = torch.randn((batch_size, M, K), device='cuda', dtype=torch.float16)
B = torch.randn((batch_size, K, N), device='cuda', dtype=torch.float16)
C = torch.zeros((batch_size, M, N), device='cuda', dtype=torch.float16)


@triton.autotune(
    configs = [triton.Config({'BLOCK_SIZE_M': bm_size, 'BLOCK_SIZE_N': bn_size, 'BLOCK_SIZE_K': bk_size, 'GROUP_SIZE_M': gp_size}, 
                            num_warps=num_warps)],
    key=['M', 'N', 'K'],
)
@triton.jit
def batch_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_a0, stride_am, stride_ak,  #
        stride_b0, stride_bk, stride_bn,  #
        stride_c0, stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    batch_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + batch_id*stride_a0 + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + batch_id*stride_b0 + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + batch_id*stride_c0 + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


grid = lambda meta: (triton.cdiv(meta['M'], meta['BLOCK_SIZE_M']) * triton.cdiv(meta['N'], meta['BLOCK_SIZE_N']), batch_size,)
batch_matmul_kernel[grid](A, B, C, M, N, K, 
                          A.stride(0), A.stride(1), A.stride(2), B.stride(0), B.stride(1), B.stride(2), C.stride(0), C.stride(1), C.stride(2), 
                        )


# warmup - 3 runs
for i in range(3):
    batch_matmul_kernel[grid](A, B, C, M, N, K, 
                        A.stride(0), A.stride(1), A.stride(2), B.stride(0), B.stride(1), B.stride(2), C.stride(0), C.stride(1), C.stride(2), 
                        )

time_array = []
for i in range(9):
    start_torch = torch.cuda.Event(enable_timing=True)
    end_torch = torch.cuda.Event(enable_timing=True)
    start_torch.record()
    batch_matmul_kernel[grid](A, B, C, M, N, K, 
                        A.stride(0), A.stride(1), A.stride(2), B.stride(0), B.stride(1), B.stride(2), C.stride(0), C.stride(1), C.stride(2), 
                        )
    end_torch.record()
    torch.cuda.synchronize()
    torch_time = start_torch.elapsed_time(end_torch)
    time_array.append(torch_time)

median_time = statistics.median(time_array)
print(f"batch_size={batch_size} M={M} N={N} K={K} bm_size={bm_size} bn_size={bn_size} bk_size={bk_size} gp_size={gp_size} num_warps={num_warps}")
print(f"Performance is {2*0.000001*batch_size*M*N*K/median_time:.0f} GFLOPs")

print("COMPLETE!\n")

