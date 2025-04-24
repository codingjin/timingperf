#!/bin/bash

python conv_test.py N=256 C=3 H=224 W=224 K=64 R=7 S=7

python conv_test.py N=256 C=3 H=224 W=224 K=64 R=7 S=7 stride=2

python conv_test.py N=256 C=3 H=224 W=224 K=64 R=7 S=7 stride=8

#python conv_test.py N=256 C=3 H=224 W=224 K=64 R=7 S=7 stride=1 padding=0 dilation=1

#python conv_test.py N=256 C=3 H=224 W=224 K=64 R=7 S=7 stride=2 padding=0 dilation=1

#python conv_test.py N=256 C=3 H=224 W=224 K=64 R=7 S=7 stride=8 padding=0 dilation=1