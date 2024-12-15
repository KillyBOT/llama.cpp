#include "esp-riscv/esp.h"
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "ggml-quants.h"
#include "ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#include "esp-acc-prints.h"
#include "esp-gemm-cfg.h"
#include "esp-gemm.h"

#define UNUSED GGML_UNUSED

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc,
 *                     0, 1,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
// bool esp_riscv_gemm(int64_t m, int64_t n, int64_t k, const void *A, int64_t lda,
//                     const void *B, int64_t ldb, void *C, int64_t ldc, int ith,
//                     int nth, int Atype, int Btype, int Ctype) {
//
//     size_t src0_len, src1_len, dest_len, src0_size, src1_size, dest_size;
//     float *accel_buf;
//     float *aqk, *bqk, *cqk;
//
//     assert(m >= 0);
//     assert(n >= 0);
//     assert(k >= 0);
//     assert(lda >= k);
//     assert(ldb >= k);
//     assert(ldc >= m);
//     assert(nth > 0); // Unused for now
//     assert(ith < nth); // Unused for now
//
//     if (n < 2)
//         return false;
//
//     if (Ctype != GGML_TYPE_F32)
//         return false;
//
//     switch (Atype) {
//     case GGML_TYPE_Q4_0: {
//         if (Btype != GGML_TYPE_Q8_0)
//             return false;
//
//         src0_len = m * k;
//         src1_len = k * n;
//         dest_len = m * n;
//         src0_size = src0_len * sizeof(float);
//         src1_size = src1_len * sizeof(float);
//         dest_size = dest_len * sizeof(float);
//
//         /* Allocate a buffer for the accelerator. For now, just make it max size */
//         // accel_buf = (token_t *)esp_alloc(MAX_SIZE);
//         // cfg_000[0].hw_buf = accel_buf;
//
//         gemm_cfg_000[0].do_relu = 0;
//         gemm_cfg_000[0].transpose = 1;
//         gemm_cfg_000[0].ninputs = 1; // TODO: Batch more inputs together?
//         gemm_cfg_000[0].d1 = m;
//         gemm_cfg_000[0].d2 = k;
//         gemm_cfg_000[0].d3 = n;
//
//         print_gemm_cfg(cfg_000, gemm_cfg_000);
//
//         // accel_buf = (float *)esp_alloc(MAX_SIZE);
//
//         // aqk = accel_buf;
//         // bqk = accel_buf + src0_len;
//
//         // int64_t ytiles = (m - m0) / RM;
//         // int64_t xtiles = (n - n0) / RN;
//         // int64_t ntiles = xtiles * ytiles;
//         // int64_t duty = (tiles + nth - 1) / nth;
//         // int64_t start = duty * ith;
//         // int64_t end = start + duty;
//         // if (end > tiles)
//         //     end = tiles;
//         // for (int64_t job = start; job < end; ++job) {
//         //     int64_t ii = m0 + job / xtiles * RM;
//         //     int64_t jj = n0 + job % xtiles * RN;
//         //     D Cv[RN][RM] = {};
//         //     for (int64_t l = 0; l < k; l += KN)
//         //         for (int64_t j = 0; j < RN; ++j)
//         //             for (int64_t i = 0; i < RM; ++i)
//         //                 Cv[j][i] = madd(load<V>(A + lda * (ii + i) + l),
//         //                                 load<V>(B + ldb * (jj + j) + l),
//         //                                 Cv[j][i]);
//         //     for (int64_t j = 0; j < RN; ++j)
//         //         for (int64_t i = 0; i < RM; ++i)
//         //             C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
//         // }
//         //
//         // esp_free(accel_buf);
//
//         return false;
//     }
//
//     default:
//         return false;
//     }
//
//     /* Stupid trick so that the compiler doesn't complain that these values are
//     * never used */
//     (void)m;
//     (void)n;
//     (void)k;
//     (void)A;
//     (void)lda;
//     (void)B;
//     (void)ldb;
//     (void)C;
//     (void)ldc;
//     (void)ith;
//     (void)nth;
//     (void)Atype;
//     (void)Btype;
//     (void)Ctype;
// }

void ggml_vec_dot_q4_0_q8_0_esp_riscv(int o, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int mn) {
    const int qk = QK4_0;
    const int nb = o / qk;

    assert(o % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(mn == 1);
#endif
    UNUSED(mn);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;
    float *accel_buf;

    // float sumf = 0;
    //
    // float x_buff[QK4_0];
    // float y_buff[QK8_0];
    //
    // for (int k = 0; k < nb; k++) {
    //     dequantize_row_q4_0(x + k, x_buff, QK4_0);
    //     dequantize_row_q8_0(y + k, y_buff, QK8_0);
    //     for (int kk = 0; kk < qk; kk++) {
    //         sumf += x_buff[kk] * y_buff[kk];
    //     }
    // }

    // for (; ib < nb; ++ib) {
    //     int sumi0 = 0;
    //     int sumi1 = 0;
    //
    //     for (int j = 0; j < qk/2; ++j) {
    //         const int lo = (x[ib].qs[j] & 0x0F) - 8;
    //         const int hi = (x[ib].qs[j] >>   4) - 8;
    //
    //         sumi0 += (lo * y[ib].qs[j]);
    //         sumi1 += (hi * y[ib].qs[j + (qk / 2)]);
    //     }
    //
    //     int sumi = sumi0 + sumi1;
    //     sumf += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
    // }

    // *s = sumf;

    gemm_cfg_000[0].do_relu = 0;
    gemm_cfg_000[0].transpose = 1;
    gemm_cfg_000[0].ninputs = 1; // TODO: Batch more inputs together?
    gemm_cfg_000[0].d1 = 1;
    gemm_cfg_000[0].d2 = o;
    gemm_cfg_000[0].d3 = 1;
    gemm_cfg_000[0].ld_offset1 = 0;
    gemm_cfg_000[0].ld_offset2 = o;
    gemm_cfg_000[0].st_offset = 2 * o;

    print_gemm_cfg(cfg_000, gemm_cfg_000);

    accel_buf = (float *)esp_alloc(MAX_SIZE);

    dequantize_row_q4_0(x, accel_buf, o);
    dequantize_row_q8_0(y, accel_buf + o, o);

    cfg_000[0].hw_buf = accel_buf;

    esp_run(cfg_000, NACC);

    *s = accel_buf[2 * o];

    // TODO: Free the buffer at the end of execution, not every time this function is called (probably via C++)
    esp_free(accel_buf);
}

void ggml_gemv_q4_0_q8_0_esp_riscv(int o, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy, int n, int m) {
    /* In general, i iterates m, j iterates n, k iterates o */

    // printf("gemv\n");

    const int qk = QK8_0; // # of values after dequantization
    const int nb = o / qk; // # of blockx4s

    assert(o % qk == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(n);
    UNUSED(m);
    UNUSED(nb);

    float x_buff[QK4_0];
    float y_buff[QK8_0];

    float sumf;

    const block_q8_0 *y = (const block_q8_0 *)vy;

    for (int i = 0; i < m; i++) {
        const block_q4_0 *x = (const block_q4_0 *)vx + (i * nb);

        sumf = 0.0;
        for (int k = 0; k < nb; k++) {

            dequantize_row_q4_0(x + (k * qk), x_buff, qk);
            dequantize_row_q8_0(y + (k * qk), y_buff, qk);

            for (int kk = 0; kk < qk; kk++) {
                sumf += x_buff[kk] + y_buff[kk];
            }
        }

        s[i] = sumf;
    }

}


void ggml_gemm_q4_0_q8_0_esp_riscv(int o, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy, int n, int m) {
    /* In general, i iterates m, j iterates n, k iterates o */

    printf("testing\n");

    const int qk = QK8_0; // # of values after dequantization per block for S
    const int nb = o / qk; // # of blocks

    assert(o % qk == 0);
    assert(n % bn == 0);
    assert(m % bm == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(n);
    UNUSED(m);
    UNUSED(nb);

    float x_buff[QK4_0];
    float y_buff[QK8_0];

    /* The destination matrix is calculated in 4x4 chunks */
    float sumf;

    for (int j = 0; j < n; j++) {
        const block_q8_0 *y = (const block_q8_0 *)vy + j;

        for (int i = 0; i < m; i++) {
            const block_q4_0 *x = (const block_q4_0 *)vx + i;

            sumf = 0.0;
            for (int k = 0; k < nb; k++) {

                dequantize_row_q4_0(x + (k * qk), x_buff, qk);
                dequantize_row_q8_0(y + (k * qk), y_buff, qk);

                for (int kk = 0; kk < qk; kk++) {
                    sumf += x_buff[kk] * y_buff[kk];
                }
            }

            s[(j * bs) + i] = sumf;
        }
    }

}

/*
 * Find X^T * Y
 * X is an m x o matrix
 * Y is a o x n matrix
 * */
void ggml_gemm_q4_0_4x4_q8_0_esp_riscv(int o, float *restrict s, size_t bs,
                             const void *restrict vx, const void *restrict vy,
                             int n, int m) {

    /* In general, i iterates m, j iterates n, k iterates o */

    const int qk = QK8_0; // # of values after dequantization per "block" of S
    const int qb = qk / 2; // # of bytes per block, before dequantization
    const int nb = o / qk; // # of blockx4s
    const int bn = 4; // # of cols per block of Y, # of cols per block of S
    const int bm = 4; // # of rows per block of X, # of rows per block of S
    const int bo = 4; // # of cols per block of X, # of rows per block of Y

    assert(o % qk == 0);
    assert(n % bn == 0);
    assert(m % bm == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(n);
    UNUSED(m);
    UNUSED(nb);
    UNUSED(bm);
    UNUSED(bn);
    UNUSED(bo);

    /* The destination matrix is calculated in 4x4 chunks */
    float sumf[bn][bm];
    int sumi;

    // For each col of Y, block-wise
    for (int j = 0; j < n / bn; j++) {

        const block_q8_0x4 *y = (const block_q8_0x4 *)vy + (j * nb);

        // For each row of X^T, block-wise
        for (int i = 0; i < m / bm; i++) {

            const block_q4_0x4 *x = (const block_q4_0x4 *)vx + (i * nb);

            /* We need to calculate S[ii][jj] */

            /* Zero out sumf */
            for (int jj = 0; jj < bn; jj++) {
                for (int ii = 0; ii < bm; ii++) {
                    sumf[jj][ii] = 0.0;
                }
            }

            // For each blockx4...
            for (int k = 0; k < nb; k++) {

                /* Calculate X[ii][kk] * Y[kk][jj] and add it to sumf */

                // For each block specifically...
                for (int b = 0; b < qb / bo; b++) {
                    // For each col of the output block...
                    for (int jj = 0; jj < bn; jj++) {
                        // For each row of the output block...
                        for (int ii = 0; ii < bm; ii++) {

                            sumi = 0;
                            // For each relevant quant of X and Y...
                            for (int kk = 0; kk < bo; ++kk) {
                                // Extract the lo and hi nibble from X^T[ii][kk][i][k] (I'm using row-major here, it's column-major)
                                const int lo = (int)(int8_t)(x[k].qs[(b * bm * bo) + (ii * bo) + kk] << 4);
                                const int hi = (int)(int8_t)(x[k].qs[(b * bm * bo) + (ii * bo) + kk] & 0xF0);

                                // And multiply by the values of Y[kk][jj][k][j] (I'm assuming row-major here, it's column-major)
                                sumi += (
                                    lo * y[k].qs[(b * bn * bo) + (jj * bo) + kk] +
                                    hi * y[k].qs[(b * bn * bo) + (jj * bo) + kk + (qb * bo)]
                                ) >> 4;
                            }

                            // Make sure to also multiply the deltas!
                            sumf[jj][ii] += sumi * GGML_FP16_TO_FP32(x[k].d[ii]) *
                                            GGML_FP16_TO_FP32(y[k].d[jj]);
                        }
                    }
                }
            }

            for (int jj = 0; jj < bn; jj++) {
                for (int ii = 0; ii < bm; ii++) {
                    s[((j * bn + jj) * bs) + (i * bm + ii)] = sumf[jj][ii];
                }
            }

        }
    }
}
