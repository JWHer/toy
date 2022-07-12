import numpy as np
import math

# Ref https://wikidocs.net/71447
# np.set_printoptions(suppress=True, precision=4)

N=8
C=[]

for i in range(N):
    row = []
    for j in range(N):
        if i==0:
            P=1
        else:
            P = math.sqrt(2)

        # Discrete Cosine Transform Type Ⅱ kernel
        # cij = P/N^(1/2)       * cos(  π/N         * (j+1/2)*i)
        cij = P/math.sqrt(N) * math.cos(math.pi / N * (j+1/2)*i)
        row.append(cij)
    C.append(row)
C = np.array(C)
print(C)

# H.264/AVC
# cij' = round(scale*cij)
# N     colum_scale     row_scale
# 4     2^13.5          2^13.5
# 8     2^14            2^14
# 16    2^14.5          2^13.5 ...
C1 = np.round(C*pow(2,14))
print(C1)

# 정확한 변환
# https://kr.mathworks.com/help/images/discrete-cosine-transform.html
# Ref https://www.geeksforgeeks.org/discrete-cosine-transform-algorithm-program/

# 라이브러리
# fftpack.dct
# Ref https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html

# DFT OpenCV
# https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
