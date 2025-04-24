import sys
import statistics
import torch
import torch.nn as nn

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
missing = [arg for arg in required_args if arg not in arguments]
if missing:
    parser.error(f"The following required arguments are missing: {', '.join(missing)}")

optional_args = {
    'stride': 1,
    'padding': 0,
    'dilation': 1
}
# Apply default values for optional args if not provided
for key, default_value in optional_args.items():
    if key not in arguments:
        arguments[key] = str(default_value)

for key in arguments:
    if arguments[key].isdigit():
        arguments[key] = int(arguments[key])

#print(arguments)
assert len(arguments) == 10

N, C, H, W = arguments['N'], arguments['C'], arguments['H'], arguments['W']
K, R, S = arguments['K'], arguments['R'], arguments['S']
stride, padding, dilation = arguments['stride'], arguments['padding'], arguments['dilation']
P, Q = H-R+1, W-S+1

print(f"N={N} C={C} H={H} W={W}")
print(f"K={K} R={R} S={S}")
print(f"P={P} Q={Q}")
print(f"stride={stride} padding={padding} dilation={dilation}")

torch.manual_seed(149)

# NCHW
input = torch.randn(size=(N, C, H, W), dtype=torch.float16, device='cuda')

# KCRS
weight_kernel = torch.randn(size=(K, C, R, S), dtype=torch.float16, device='cuda')

# NKPQ
output = torch.zeros(size=(N, K, P, Q), dtype=torch.float16, device='cuda')

# warm-up 3 rounds
for i in range(3):
    output = nn.functional.conv2d(input, weight_kernel, stride=stride, padding=padding, dilation=dilation)

# 9 rounds
time_array = []
for i in range(9):
    start_torch = torch.cuda.Event(enable_timing=True)
    end_torch = torch.cuda.Event(enable_timing=True)
    start_torch.record()
    output = nn.functional.conv2d(input, weight_kernel, stride=stride, padding=padding, dilation=dilation)
    end_torch.record()
    torch.cuda.synchronize()
    torch_time = start_torch.elapsed_time(end_torch)
    time_array.append(torch_time)
    
median_time = statistics.median(time_array)
print(f"Pytorch conv Performance is {2*0.000001*N*K*P*Q*C*R*S/median_time:.0f} GFLOPs")

print("Complete!")
print()