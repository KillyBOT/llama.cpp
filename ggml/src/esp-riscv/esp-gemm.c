#include "esp-riscv/esp.h"
#include "ggml-impl.h"
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "ggml-quants.h"
#include "ggml-cpu-impl.h"

#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#include "esp-acc-prints.h"
#include "esp-gemm-cfg.h"
#include "esp-gemm.h"

#define UNUSED GGML_UNUSED

#define RM 8
#define RN 8

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
//         // gemm_cfg_000[0].do_relu = 0;
//         // gemm_cfg_000[0].transpose = 1;
//         // gemm_cfg_000[0].ninputs = 1; // TODO: Batch more inputs together?
//         // gemm_cfg_000[0].d1 = m;
//         // gemm_cfg_000[0].d2 = k;
//         // gemm_cfg_000[0].d3 = n;
//         //
//         // print_gemm_cfg(cfg_000, gemm_cfg_000);
//         //
//         // accel_buf = (float *)esp_alloc(MAX_SIZE);
//
//         // aqk = accel_buf;
//         // bqk = accel_buf + src0_len;
//
//         int64_t m_tiles = m / RM;
//         int64_t n_tiles = n / RN;
//         int64_t tiles = n_tiles * m_tiles;
//         int64_t duty = (tiles + nth - 1) / nth;
//         int64_t start_tile = duty * ith;
//         int64_t end_tile = start_tile + duty;
//         if (end_tile > tiles)
//             end_tile = tiles;
//         for (int64_t tile = start_tile; tile < end_tile; ++tile) {
//             int64_t i = tile / n_tiles * RM;
//             int64_t j = tile % n_tiles * RN;
//             float CC[RN][RM] = {};
//             for (int64_t l = 0; l < k; ++l) {
//                 for (int64_t jj = 0; jj < RN; ++jj) {
//                     for (int64_t ii = 0; ii < RM; ++ii) {
//                         CC[jj][ii] =
//                     }
//                 }
//             }
//
//             for (int64_t jj = 0; jj < RN; ++jj) {
//                 for (int64_t ii = 0; ii < RM; ++ii) {
//                     ((float *)C)[(ldc * (jj + j)) + (ii + i)] = CC[j][i];
//                 }
//             }
//         }
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

/* Just a copy of the hardware accelerator's GEMM procedure, to verify everything works */
/* On the actual hardware, this should not be used */
static void sw_run(int *acc_buff)
{
    esp_token_t accum;

    const unsigned int jobs = gemm_cfg_000->ninputs;
    const unsigned int m = gemm_cfg_000->d1;
    const unsigned int o = gemm_cfg_000->d2;
    const unsigned int n = gemm_cfg_000->d3;
    const unsigned int transpose = gemm_cfg_000->transpose;
    esp_token_t *As = &acc_buff[gemm_cfg_000->ld_offset1];
    esp_token_t *Bs = &acc_buff[gemm_cfg_000->ld_offset2];
    esp_token_t *Cs = &acc_buff[gemm_cfg_000->st_offset];

    for (unsigned int job = 0; job < jobs; ++job)
    {
        esp_token_t *A = &As[job * m * o];
        esp_token_t *B = &Bs[job * o * n];
        esp_token_t *C = &Cs[job * m * n];

        for (unsigned int i = 0; i < m; ++i)
        {
            for (unsigned int j = 0; j < n; ++j)
            {
                accum = 0.0;

                for (unsigned int k = 0; k < o; ++k)
                {
                    const int Aik = (i * o) + k;
                    const int Bkj = transpose ? (j * o) + k : (k * n) + j;

                    accum += A[Aik] * B[Bkj];
                }

                C[(i * n) + j] = accum;
            }
        }
    }
}

void ggml_vec_dot_q4_0_q8_0_esp_riscv(int o, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int mn) {
    const int qk = QK8_0;
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
    esp_token_t *acc_buff;
    esp_token_t *acc_x;
    esp_token_t *acc_y;
    esp_token_t *acc_i;
    esp_token_t *acc_s;

    int sum_i; // Intermediate sum
    float sum_f = 0.0; // Final sum

    // float xb[QK4_0];
    // float yb[QK8_0];
    //
    // for (int k = 0; k < nb; k++) {
    //     dequantize_row_q4_0(x + k, xb, QK4_0);
    //     dequantize_row_q8_0(y + k, yb, QK8_0);
    //     for (int kk = 0; kk < qk; kk++) {
    //         sum_f += xb[kk] * yb[kk];
    //     }
    // }
    // *s = sum_f;

    // for (k = 0; k < nb; ++k) {
    //     sum_i = 0;
    //
    //     for (kk = 0; kk < qk / 2; ++kk) {
    //         const int lo = (x[k].qs[kk] & 0x0F) - 8;
    //         const int hi = (x[k].qs[kk] >>   4) - 8;
    //
    //         sum_i += (lo * y[k].qs[kk]) + (hi * y[k].qs[kk + (qk / 2)]);
    //     }
    //
    //     sum_f += sum_i * GGML_FP16_TO_FP32(x[k].d) * GGML_FP16_TO_FP32(y[k].d);
    // }

    acc_buff = esp_alloc(sizeof(esp_token_t) * (o + o + nb + nb + 1));
    cfg_000->hw_buf = acc_buff;
    // acc_buff = (esp_token_t *)malloc(sizeof(esp_token_t) * (o + o + nb + nb + 1));
    acc_x = acc_buff;
    acc_y = acc_x + o;
    acc_i = acc_y + o;
    acc_s = acc_i + nb;

    for (int k = 0; k < nb; ++k) {
        for (int kk = 0; kk < qk / 2; ++kk) {
            acc_x[(k * qk) + kk] =            (x[k].qs[kk] & 0x0F) - 8;
            acc_x[(k * qk) + kk + (qk / 2)] = (x[k].qs[kk] >> 4) - 8;
            acc_y[(k * qk) + kk] =             y[k].qs[kk];
            acc_y[(k * qk) + kk + (qk / 2)] =  y[k].qs[kk + (qk / 2)];
        }
    }

    /* Find the dot products of each block */
    gemm_cfg_000[0].transpose = 1; // Technically not needed, but still nice to include
    gemm_cfg_000[0].ninputs = nb; // Number of inputs == number of blocks
    gemm_cfg_000[0].d1 = 1;
    gemm_cfg_000[0].d2 = qk;
    gemm_cfg_000[0].d3 = 1;
    gemm_cfg_000[0].ld_offset1 = 0;
    gemm_cfg_000[0].ld_offset2 = o;
    gemm_cfg_000[0].st_offset = o + o;

    // sw_run(acc_buff);
    esp_run(cfg_000, NACC);

    /* Then, find the dot product of each block's dot product with their deltas */
    /* Not only do floats lose precision, the number of blocks is usually quite low, so it's probably okay to just do this */
    for (int k = 0; k < nb; k++) {
        sum_f += acc_i[k] * GGML_FP16_TO_FP32(x[k].d) * GGML_FP16_TO_FP32(y[k].d);
    }
    *s = sum_f;

    // printf("X:[");
    // for (int k = 0; k < o; k++) {
    //     printf(k == 0 ? "%d" : " %d", acc_x[k]);
    // }
    // printf("]\nY:[");
    // for (int k = 0; k < o; k++) {
    //     printf(k == 0 ? "%d" : " %d", acc_y[k]);
    // }
    // printf("]\nS:[");
    // for (int k = 0; k < nb; k++) {
    //     printf(k == 0 ? "%d" : " %d", acc_y[o + k]);
    // }
    // printf("]\nD:[");
    // for (int k = 0; k < nb; k++) {
    //     printf(k == 0 ? "%f" : " %f", fx2float(acc_d[k], FX_IL));
    // }
    // printf("]\n");
    // printf("Final result: %f\n", sum_f);

    // ggml_vec_dot_q4_0_q8_0(o, &sum_f, bs, vx, bx, vy, by, mn);
    // printf("Expected_result: %f\n", sum_f);

    // free(acc_buff);
    esp_free(acc_buff);
}

void ggml_gemv_q4_0_q8_0_esp_riscv(int o, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy, int n, int m) {
    /* In general, i iterates m, j iterates n, k iterates o */

    const int qk = QK8_0;
    const int nb = o / qk; // # of blocks

    assert(o % qk == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(n);
    UNUSED(m);
    UNUSED(nb);

    // float xb[QK4_0];
    // float y_buff[QK8_0];
    // float *xb = malloc(sizeof(float) * o);
    // float *yb = malloc(sizeof(float) * o);
    //
    // assert(xb != NULL);
    // assert(yb != NULL);

    int sumi_lo;
    int sumi_hi;
    float sumf;

    const block_q8_0 *y = (const block_q8_0 *)vy;

    // for (int b = 0; b < nb; b++) {
    //     const float d = GGML_FP16_TO_FP32(y[b].d);
    //     if (d < 1.0)
    //         printf("%d: %f\n", b, d);
    // }

    // dequantize_row_q8_0(y, yb, o);

    // for (int k = 0; k < o; k++) {
    //     if (k == 0) printf("[0:%f", yb[k]);
    //     else if (k == (o - 1)) printf(", %d:%f]\n", o-1, yb[k]);
    //     else printf(", %d:%f", k, yb[k]);
    // }
    //
    // for (int b = 0; b < nb; b++) {
    //     printf("%f\n", GGML_FP16_TO_FP32(y[b].d));
    // }
    //

    for (int i = 0; i < m; i++) {
        const block_q4_0 *x = (const block_q4_0 *)vx + (i * nb);

        sumf = 0.0;

        for (int k = 0; k < nb; k++) {

            sumi_lo = 0;
            sumi_hi = 0;
            for (int kk = 0; kk < qk; kk++) {
                const int lo = (x[k].qs[kk] & 0x0F) - 8;
                const int hi = (x[k].qs[kk] >>   4) - 8;

                sumi_lo += lo * y[k].qs[kk];
                sumi_hi += hi * y[k].qs[kk + (qk / 2)];
            }
            sumf += (sumi_lo + sumi_hi) * GGML_FP16_TO_FP32(x[k].d) * GGML_FP16_TO_FP32(y[k].d);
        }

        s[i] = sumf;
    }

    // free(xb);
    // free(yb);

}


void ggml_gemm_q4_0_q8_0_esp_riscv(int o, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy, int n, int m) {
    /* In general, i iterates m, j iterates n, k iterates o */

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

    // float xb[QK4_0];
    // float yb[QK8_0];

    /* The destination matrix is calculated in 4x4 chunks */
    int sumi_lo;
    int sumi_hi;
    float sumf;

    for (int j = 0; j < n; j++) {
        const block_q8_0 *y = (const block_q8_0 *)vy + (j * nb);

        for (int i = 0; i < m; i++) {
            const block_q4_0 *x = (const block_q4_0 *)vx + (i * nb);

            sumf = 0.0;
            for (int k = 0; k < nb; k++) {

                sumi_lo = 0;
                sumi_hi = 0;
                for (int kk = 0; kk < qk; kk++) {
                    const int lo = (x[k].qs[kk] & 0x0F) - 8;
                    const int hi = (x[k].qs[kk] >>   4) - 8;

                    sumi_lo += lo * y[k].qs[kk];
                    sumi_hi += hi * y[k].qs[kk + (qk / 2)];
                }
                sumf += (sumi_lo + sumi_hi) * GGML_FP16_TO_FP32(x[k].d) * GGML_FP16_TO_FP32(y[k].d);
            }

            s[(j * bs) + i] = sumf;
        }
    }

}
//
// void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int64_t k) {
//     static const int qk = QK4_0;
//
//     assert(k % qk == 0);
//
//     const int nb = k / qk;
//
//     for (int i = 0; i < nb; i++) {
//         const float d = GGML_FP16_TO_FP32(x[i].d);
//
//         for (int j = 0; j < qk/2; ++j) {
//             const int x0 = (x[i].qs[j] & 0x0F) - 8;
//             const int x1 = (x[i].qs[j] >>   4) - 8;
//
//             y[i*qk + j + 0   ] = x0*d;
//             y[i*qk + j + qk/2] = x1*d;
//         }
//     }
// }

/*
 * Find X^T * Y
 * X is an m x o matrix
 * Y is a o x n matrix
 * */
void ggml_gemv_q4_0_4x4_q8_0_esp_riscv(int o, float *restrict s, size_t bs,
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
    UNUSED(bo);

    /* The destination matrix is calculated in 4x4 chunks */
    float sumf[bm];
    int sumi;

    const block_q8_0x4 *y = (const block_q8_0x4 *)vy;

    // For each row of X^T, block-wise
    for (int i = 0; i < m / bm; i++) {

        const block_q4_0x4 *x = (const block_q4_0x4 *)vx + (i * nb);

        /* We need to calculate S[ii][jj] */
        /* Zero out sumf */
        for (int ii = 0; ii < bm; ii++) {
            sumf[ii] = 0.0;
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
                        sumf[ii] += sumi * GGML_FP16_TO_FP32(x[k].d[ii]) * GGML_FP16_TO_FP32(y[k].d[jj]);
                    }
                }
            }
        }

        for (int ii = 0; ii < bm; ii++) {
            s[(i * bm) + ii] = sumf[ii];
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
