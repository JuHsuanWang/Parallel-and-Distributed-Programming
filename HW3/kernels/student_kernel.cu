#include "../math_utils.h"
#include <stdio.h>

// =============================================================================
// HW4: CUDA Matmul Optimization
//
// Version 2: 2D register-tiled SGEMM
//
// Block tile: 64 x 64 output elements
// K tile:     8
// Threads:    16 x 16 = 256 threads per block
// Each thread computes a 4 x 4 micro-tile in registers.
//
// Matrix layout is row-major.
// C = alpha * (A @ B) + beta * C
// =============================================================================

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15
    int tid = ty * blockDim.x + tx;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int row_base = block_row + ty * TM;
    int col_base = block_col + tx * TN;

    float acc[TM][TN];

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile: BM x BK = 64 x 8 = 512 elements
        for (int idx = tid; idx < BM * BK; idx += 256) {
            int r = idx / BK;
            int c = idx % BK;

            int global_r = block_row + r;
            int global_c = k0 + c;

            if (global_r < M && global_c < K) {
                As[r][c] = A[global_r * K + global_c];
            } else {
                As[r][c] = 0.0f;
            }
        }

        // Load B tile: BK x BN = 8 x 64 = 512 elements
        for (int idx = tid; idx < BK * BN; idx += 256) {
            int r = idx / BN;
            int c = idx % BN;

            int global_r = k0 + r;
            int global_c = block_col + c;

            if (global_r < K && global_c < N) {
                Bs[r][c] = B[global_r * N + global_c];
            } else {
                Bs[r][c] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[ty * TM + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[k][tx * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = row_base + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = col_base + j;

            if (row < M && col < N) {
                C[row * N + col] =
                    alpha * acc[i][j] + beta * C[row * N + col];
            }
        }
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
