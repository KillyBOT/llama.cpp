#include "ggml-esp/esp.h"
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
// bool esp_riscv_gemm(int64_t m, int64_t n, int64_t o, const void *vA, int64_t lda,
//                     const void *vB, int64_t ldb, void *vC, int64_t ldc, int ith,
//                     int nth, int Atype, int Btype, int Ctype) {
//
//     size_t src0_len, src1_len, dest_len, src0_size, src1_size, dest_size;
//     float *acc_buff;
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
//         src0_len = m * o;
//         src1_len = o * n;
//         dest_len = m * n;
//         src0_size = src0_len * sizeof(float);
//         src1_size = src1_len * sizeof(float);
//         dest_size = dest_len * sizeof(float);
//
//         const block_q4_0 *restrict A = vA;
//         const block_q8_0 *restrict B = vB;
//         float *restrict C = vC;
//         // const int qk = QK8_0;
//         float sum_f;
//
//         float A_buff[o * QK4_0];
//         float B_buff[o * QK8_0];
//
//         // for (int k = 0; k < o; k++) {
//         //     sum_f = 0.0;
//         //     for (int j = 0; j < n; j++) {
//         //         for (int i = 0; i < m; i++) {
//         //             dequantize_row_q4_0(A + (lda * i) + k, A_buff, QK4_0);
//         //             dequantize_row_q8_0(B + (ldb * j) + k, B_buff, QK8_0);
//         //
//         //         }
//         //     }
//         // }
//
//         for (int j = 0; j < n; j++) {
//             const block_q8_0 *restrict Bj = B + (j * ldb);
//             dequantize_row_q8_0(Bj, B_buff, o * QK8_0);
//
//             for (int i = 0; i < m; i++) {
//                 const block_q4_0 *restrict Ai = A + (i * lda);
//
//                 dequantize_row_q4_0(Ai, A_buff, o * QK4_0);
//                 //
//                 // printf("Ai: [");
//                 // for (int k = 0; k < o; k++) {
//                 //     printf(k == 0 ? "%f" : " %f", A_buff[k]);
//                 // }
//                 // printf("]\n");
//
//                 sum_f = 0.0;
//                 for (int k = 0; k < o * QK8_0; k++) {
//                     sum_f += A_buff[k] * B_buff[k];
//                 }
//                 // for (int k = 0; k < o / qk; k++) {
//                 //
//                 //     sumi_lo = 0;
//                 //     sumi_hi = 0;
//                 //     for (int kk = 0; kk < qk; kk++) {
//                 //         const int lo = (Ai[k].qs[kk] & 0x0F) - 8;
//                 //         const int hi = (Ai[k].qs[kk] >>   4) - 8;
//                 //
//                 //         sumi_lo += lo * Bj[k].qs[kk];
//                 //         sumi_hi += hi * Bj[k].qs[kk + (qk / 2)];
//                 //     }
//                 //     sum_f += (sumi_lo + sumi_hi) * GGML_FP16_TO_FP32(Ai[k].d) * GGML_FP16_TO_FP32(Bj[k].d);
//                 // }
//
//                 C[(j * ldc) + i] = sum_f;
//             }
//         }
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
//         // int64_t m_tiles = m / RM;
//         // int64_t n_tiles = n / RN;
//         // int64_t tiles = n_tiles * m_tiles;
//         // int64_t duty = (tiles + nth - 1) / nth;
//         // int64_t start_tile = duty * ith;
//         // int64_t end_tile = start_tile + duty;
//         // if (end_tile > tiles)
//         //     end_tile = tiles;
//         // for (int64_t tile = start_tile; tile < end_tile; ++tile) {
//         //     int64_t i = (tile / n_tiles) * RM;
//         //     int64_t j = (tile % n_tiles) * RN;
//         //     float CC[RN][RM] = {};
//         //     for (int64_t k = 0; k < o; ++k) {
//         //         for (int64_t jj = 0; jj < RN; ++jj) {
//         //             for (int64_t ii = 0; ii < RM; ++ii) {
//         //                 CC[jj][ii] =
//         //             }
//         //         }
//         //     }
//         //
//         //     for (int64_t jj = 0; jj < RN; ++jj) {
//         //         for (int64_t ii = 0; ii < RM; ++ii) {
//         //             ((float *)C)[(ldc * (jj + j)) + (ii + i)] = CC[j][i];
//         //         }
//         //     }
//         // }
//         //
//         // esp_free(accel_buf);
//
//         return true;
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
//     (void)o;
//     (void)vA;
//     (void)lda;
//     (void)vB;
//     (void)ldb;
//     (void)vC;
//     (void)ldc;
//     (void)ith;
//     (void)nth;
//     (void)Atype;
//     (void)Btype;
//     (void)Ctype;
// }

/* Just a copy of the hardware accelerator's GEMM procedure, to verify everything works */
/* On the actual hardware, this should not be used */
/* The inlining is stupid I know, but since this is called so frequently it makes sense to */
static inline void sw_run(esp_thread_info_t cfg[])
{
    esp_token_t accum;

    const unsigned int jobs = gemm_cfg_000->ninputs;
    const unsigned int m = gemm_cfg_000->d1;
    const unsigned int k = gemm_cfg_000->d2;
    const unsigned int n = gemm_cfg_000->d3;
    const unsigned int transpose = gemm_cfg_000->transpose;

    esp_token_t *hw_buf = cfg[0].hw_buf;
    const esp_token_t *As = &hw_buf[gemm_cfg_000->ld_offset1];
    const esp_token_t *Bs = &hw_buf[gemm_cfg_000->ld_offset2];
    esp_token_t *Cs = &hw_buf[gemm_cfg_000->st_offset];

    for (unsigned int job = 0; job < jobs; ++job)
    {
        const esp_token_t *A = &As[job * m * k];
        const esp_token_t *B = &Bs[job * k * n];
        esp_token_t *C = &Cs[job * m * n];

        for (unsigned int i = 0; i < m; ++i)
        {
            for (unsigned int j = 0; j < n; ++j)
            {
                accum = 0.0;

                for (unsigned int l = 0; l < k; ++l)
                {
                    const int Aik = (i * k) + l;
                    const int Bkj = transpose ? (j * k) + l : (l * n) + j;

                    accum += A[Aik] * B[Bkj];
                }

                C[(i * n) + j] = accum;
            }
        }
    }
}

void ggml_vec_dot_q4_0_q8_0_esp(int k, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int mn) {
    const int qk = QK8_0;
    const int n_b = k / qk;

    assert(k % qk == 0);
    assert(mn == 1);

    UNUSED(mn);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    /* The accelerator's buffer. Contains enough room for the input vectors and the output vector*/
#if defined(GGML_USE_ESP_TEST)
    esp_token_t *acc_buff = (esp_token_t *)malloc(sizeof(esp_token_t) * (k + k + n_b));
#elif defined(GGML_USE_ESP_RISCV)
    printf("esp_alloc...");
    esp_token_t *acc_buff = esp_alloc(sizeof(esp_token_t) * (k + k + n_b));
    printf("finished\n");
#endif
    esp_token_t *acc_x = acc_buff; /* Location of X */
    esp_token_t *acc_y = acc_buff + k; /* Location of Y */
    esp_token_t *acc_i = acc_buff + (2 * k); /* Location of X * Y, before using deltas */

    float sum_f = 0.0; // Final sum; used to reduce the number of times s needs to be dereferenced

    // For each block...
    for (int b = 0; b < n_b; ++b) {
        // For each quant
        for (int q = 0; q < qk / 2; ++q) {
            // Extract low nibble
            acc_x[(b * qk) + q] =            (x[b].qs[q] & 0x0F) - 8;
            // Extract high nibble
            acc_x[(b * qk) + q + (qk / 2)] = (x[b].qs[q] >>   4) - 8;
        }
        for (int q = 0; q < qk; ++q) {
            // Extract byte
            acc_y[(b * qk) + q] = (esp_token_t)y[b].qs[q];
        }
    }

    /* Find the dot products of each block */
    cfg_000->hw_buf = acc_buff;
    gemm_cfg_000->transpose = 1; // Technically not needed, but still nice to include
    gemm_cfg_000->ninputs = n_b; // We are finding n_b dot products at a time
    gemm_cfg_000->d1 = 1;
    gemm_cfg_000->d2 = qk; // Each dot product takes qk elements
    gemm_cfg_000->d3 = 1;
    gemm_cfg_000->ld_offset1 = 0;
    gemm_cfg_000->ld_offset2 = k;
    gemm_cfg_000->st_offset = 2 * k;

#if defined(GGML_USE_ESP_TEST)
    sw_run(cfg_000);
#elif defined(GGML_USE_ESP_RISCV)
    printf("esp_run...");
    print_gemm_cfg(cfg_000, gemm_cfg_000);
    esp_run(cfg_000, NACC);
    printf("finished\n");
#endif

    /* Then, find the dot product of each block's dot product with their deltas */
    /* Not only do floats lose precision with how the accelerator is currently designed, there are 32 elements per each block, meaning it's not the end of the world to just multiply deltas here */
    for (int l = 0; l < n_b; l++) {
        sum_f += acc_i[l] * GGML_FP16_TO_FP32(x[l].d) * GGML_FP16_TO_FP32(y[l].d);
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
    //
    // ggml_vec_dot_q4_0_q8_0(o, &sum_f, bs, vx, bx, vy, by, mn);
    // printf("Expected_result: %f\n", sum_f);

#if defined(GGML_USE_ESP_TEST)
    free(acc_buff);
#elif defined(GGML_USE_ESP_RISCV)
    printf("esp_free...");
    esp_free(acc_buff);
    printf("finished\n");
#endif
}

// TODO: Make this not really slow...
void ggml_vec_dot_q4_K_q8_K_esp(int k, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int mn) {

    assert(k % QK_K == 0);
    assert(mn == 1);

    UNUSED(mn);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int n_sb = k / QK_K; /* Number of superblocks */
    const int qk   = 32; /* Number of dequantized elements per block */
    const int n_b  = 8; /* Number of blocks per superblock */
    const int n_ub = 4;  /* Number of sub-blocks per block */

    static const uint32_t scales_mins_mask_1 = 0x3f3f3f3f;
    static const uint32_t scales_mins_mask_2 = 0x0f0f0f0f;
    static const uint32_t scales_mins_mask_3 = 0x03030303;

    float sum_f = 0.0; // Final sum; used to reduce the number of times s needs to be dereferenced

    uint32_t scales_mins[4];
    const uint8_t *scales = (const uint8_t*)&scales_mins[0];
    const uint8_t *mins   = (const uint8_t*)&scales_mins[2];

    int8_t  x_q8[QK_K]; // Quants of superblock X
    float   sums[n_b];  // Sums of each block

#if defined(GGML_USE_ESP_TEST)
    esp_token_t *acc_buff = (esp_token_t *)malloc(sizeof(esp_token_t) * (k + k + n_b * n_b * n_sb + n_b * n_b * n_sb + n_b * n_sb));
#elif defined (GGML_USE_ESP_RISCV)
    esp_token_t *acc_buff = (esp_token_t *)esp_alloc(sizeof(esp_token_t) * (k + k + n_b * n_b * n_sb + n_b * n_b * n_sb + n_b * n_sb));
#endif
    esp_token_t *acc_x = acc_buff;
    esp_token_t *acc_y = acc_x + k;
    esp_token_t *acc_unscaled = acc_y + k;
    esp_token_t *acc_scales = acc_unscaled + (n_b * n_b * n_sb);
    esp_token_t *acc_scaled = acc_scales + (n_b * n_b * n_sb);

    /* For each superblock...*/
    for (int sb = 0; sb < n_sb; ++sb) {

        /* Fetch the current scales and mins and dequantize */
        memcpy(scales_mins, x[sb].scales, K_SCALE_SIZE);
        scales_mins[3] = ((scales_mins[2] >> 4) & scales_mins_mask_2) | (((scales_mins[1] >> 6) & scales_mins_mask_3) << 4);
        const uint32_t tmp = scales_mins[1] & scales_mins_mask_1;
        scales_mins[1] = (scales_mins[2] & scales_mins_mask_2) | (((scales_mins[0] >> 6) & scales_mins_mask_3) << 4);
        scales_mins[2] = tmp;
        scales_mins[0] &= scales_mins_mask_1;

        /* Extract quants of X */
        const uint8_t * restrict x_q4 = x[sb].qs;
        for (int b = 0; b < n_b / 2; b++) {
            for (int q = 0; q < qk; ++q) {
                x_q8[(b * qk * 2) + q]      = (int8_t)(x_q4[(b * qk) + q] & 0xF);
                x_q8[(b * qk * 2) + q + qk] = (int8_t)(x_q4[(b * qk) + q] >>  4);
            }
        }

        const int8_t *restrict y_q8 = y[sb].qs;
        for (int b = 0; b < n_b; ++b) {
            for (int ub = 0; ub < n_ub; ub++) {
                /* Then, multiply by scales */
                for (int d = 0; d < n_b; ++d) {
                    acc_x[(sb * QK_K) + (d * qk) + (b * n_ub) + ub] = (esp_token_t)x_q8[(b * qk) + (ub * n_b) + d];
                    acc_y[(sb * QK_K) + (d * qk) + (b * n_ub) + ub] = (esp_token_t)y_q8[(b * qk) + (ub * n_b) + d];
                }
            }
            for (int d = 0; d < n_b; d++) {
                acc_scales[(sb * n_b * n_b) + (d * n_b) + b] = scales[b];
            }
        }

        /* Subtract min deltas while we are aware of the mins */
        int sums_b = 0;
        for (int b = 0; b < n_b * 2; ++b) {
            sums_b += y[sb].bsums[b] * mins[b/2];
        }
        const float delta_min = GGML_FP16_TO_FP32(x[sb].dmin) * y[sb].d;
        sum_f -= delta_min * sums_b;
    }

    // printf("First iter:[");
    // for (int l = 0; l < k; l++) {
    //     printf("(%d * %d) ", acc_x[l], acc_y[l]);
    // }
    // printf("]\n");

    /* First run: Calculate the non-scaled dot products */
    cfg_000->hw_buf = acc_buff;
    gemm_cfg_000->transpose = 1; // Technically not needed, but still nice to include
    gemm_cfg_000->ninputs = n_b * n_b * n_sb; // # of inputs = # of blocks * # of dot products
    gemm_cfg_000->d1 = 1;
    gemm_cfg_000->d2 = n_ub;
    gemm_cfg_000->d3 = 1;
    gemm_cfg_000->ld_offset1 = 0;
    gemm_cfg_000->ld_offset2 = k;
    gemm_cfg_000->st_offset  = k * 2;

#if defined(GGML_USE_ESP_TEST)
    sw_run(cfg_000);
#elif defined(GGML_USE_ESP_RISCV)
    esp_run(cfg_000, NACC);
#endif

    // printf("Second iter:[");
    // for (int l = 0; l < n_b * n_b * n_sb; l++) {
    //     printf("(%d * %d) ", acc_unscaled[l], acc_scales[l]);
    // }
    // printf("]\n");


    /* Second run: Calculate the scaled dot products */
    gemm_cfg_000->transpose = 1;
    gemm_cfg_000->ninputs = n_b * n_sb;
    gemm_cfg_000->d1 = 1;
    gemm_cfg_000->d2 = n_b;
    gemm_cfg_000->d3 = 1;
    gemm_cfg_000->ld_offset1 = k * 2;
    gemm_cfg_000->ld_offset2 = k * 2 + n_b * n_b * n_sb;
    gemm_cfg_000->st_offset  = k * 2 + n_b * n_b * n_sb * 2;

    sw_run(cfg_000);

    /*printf("Scaled dot products:[");*/
    /*for (int l = 0; l < n_b * n_sb; l++) {*/
    /*    printf("%d (%d) ", acc_scaled[l], dots[l]);*/
    /*}*/
    /*printf("]\n");*/

    /* Set all sums to 0 */
    memset(sums, 0, n_b*sizeof(float));

    for (int sb = 0; sb < n_sb; ++sb) {

        /* Multiply answers by deltas */
        const float delta = GGML_FP16_TO_FP32(x[sb].d) * y[sb].d;
        for (int b = 0; b < n_b; ++b) {
            sums[b] += delta * acc_scaled[(sb * n_b) + b];
        }
    }

    for (int b = 0; b < n_b; ++b) {
        sum_f += sums[b];
    }

    *s = sum_f;

#if defined(GGML_USE_ESP_TEST)
    free(acc_buff);
#elif defined(GGML_USE_ESP_RISCV)
    esp_free(acc_buff);
#endif
    // printf("Finished\n");
}

// TODO: Utilize the accelerator here
void ggml_vec_dot_q6_K_q8_K_esp_riscv(int o, float *restrict s, size_t bs, const void *restrict vx, size_t bx, const void *restrict vy, size_t by, int mn) {
    assert(o % QK_K == 0);
    assert(mn == 1);
    UNUSED(mn);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int n_sb = o / QK_K;
    // const int n_b = QK_K / 32;

    int8_t  x_qs[QK_K];
    int16_t dots_unscaled[8];
    int32_t dots_scaled[8];
    float   sums [8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < n_sb; ++i) {
        const uint8_t * restrict x_ql = x[i].ql;
        const uint8_t * restrict x_qh = x[i].qh;
        const  int8_t * restrict y_qs = y[i].qs;

        int8_t * restrict x_qs_it = x_qs;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                x_qs_it[l +  0] = (int8_t)((x_ql[l +  0] & 0xF) | (((x_qh[l] >> 0) & 3) << 4)) - 32;
                x_qs_it[l + 32] = (int8_t)((x_ql[l + 32] & 0xF) | (((x_qh[l] >> 2) & 3) << 4)) - 32;
                x_qs_it[l + 64] = (int8_t)((x_ql[l +  0] >>  4) | (((x_qh[l] >> 4) & 3) << 4)) - 32;
                x_qs_it[l + 96] = (int8_t)((x_ql[l + 32] >>  4) | (((x_qh[l] >> 6) & 3) << 4)) - 32;
            }
            x_qs_it += 128;
            x_ql += 64;
            x_qh += 32;
        }

        memset(dots_scaled, 0, 8*sizeof(int32_t));
        x_qs_it = x_qs;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[j];
            for (int l = 0; l < 8; ++l) dots_unscaled[l] = y_qs[l] * x_qs_it[l];
            for (int l = 0; l < 8; ++l) dots_scaled[l] += scale * dots_unscaled[l];
            y_qs += 8; x_qs_it += 8;
            for (int l = 0; l < 8; ++l) dots_unscaled[l] = y_qs[l] * x_qs_it[l];
            for (int l = 0; l < 8; ++l) dots_scaled[l] += scale * dots_unscaled[l];
            y_qs += 8; x_qs_it += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * dots_scaled[l];
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

}

void ggml_vec_dot_q8_0_q8_0_esp(int k, float *restrict s, size_t bs,
                                      const void *restrict vx, size_t bx,
                                      const void *restrict vy, size_t by,
                                      int mn) {

    const int qk = QK8_0;
    const int n_b = k / qk;

    assert(k % qk == 0);
    assert(mn == 1);

    UNUSED(mn);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    /* The accelerator's buffer. Contains enough room for the input vectors and the output vector*/
#if defined(GGML_USE_ESP_TEST)
    esp_token_t *acc_buff = (esp_token_t *)malloc(sizeof(esp_token_t) * (k + k + n_b));
#elif defined(GGML_USE_ESP_RISCV)
    esp_token_t *acc_buff = esp_alloc(sizeof(esp_token_t) * (k + k + n_b));
#endif
    esp_token_t *acc_x = acc_buff; /* Location of X */
    esp_token_t *acc_y = acc_buff + k; /* Location of Y */
    esp_token_t *acc_i = acc_buff + (2 * k); /* Location of X * Y, before using deltas */

    float sum_f = 0.0; // Final sum; used to reduce the number of times s needs to be dereferenced

    for (int l = 0; l < n_b; ++l) {
        for (int kk = 0; kk < qk; ++kk) {
            acc_x[(l * qk) + kk] = (esp_token_t)x[l].qs[kk];
            acc_y[(l * qk) + kk] = (esp_token_t)y[l].qs[kk];
        }
    }

    /* Find the dot products of each block */
    cfg_000->hw_buf = acc_buff;
    gemm_cfg_000->transpose = 1; // Technically not needed, but still nice to include
    gemm_cfg_000->ninputs = n_b; // We are finding n_b dot products at a time
    gemm_cfg_000->d1 = 1;
    gemm_cfg_000->d2 = qk; // Each dot product takes qk elements
    gemm_cfg_000->d3 = 1;
    gemm_cfg_000->ld_offset1 = 0;
    gemm_cfg_000->ld_offset2 = k;
    gemm_cfg_000->st_offset = 2 * k;

#if defined(GGML_USE_ESP_TEST)
    sw_run(cfg_000);
#elif defined(GGML_USE_ESP_RISCV)
    esp_run(cfg_000, NACC);
#endif

    /* Then, find the dot product of each block's dot product with their deltas */
    /* Not only do floats lose precision with how the accelerator is currently designed, there are 32 elements per each block, meaning it's not the end of the world to just multiply deltas here */
    for (int l = 0; l < n_b; l++) {
        sum_f += acc_i[l] * GGML_FP16_TO_FP32(x[l].d) * GGML_FP16_TO_FP32(y[l].d);
    }

    *s = sum_f;

#if defined(GGML_USE_ESP_TEST)
    free(acc_buff);
#elif defined(GGML_USE_ESP_RISCV)
    esp_free(acc_buff);
#endif
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

    int sum_i;
    float sum_f;

    const block_q8_0 * restrict y = (const block_q8_0 *)vy;

    // printf("Y: [");
    // for (int k = 0; k < nb; k++) {
    //     printf("\t%d [%f] [", k, y[k].d);
    //     for (int kk = 0; kk < qk; kk++) {
    //         printf(kk == 0 ? "%d" : " %d", y[k].qs[kk]);
    //     }
    //     printf("]\n");
    // }
    // printf("]\n");

    for (int i = 0; i < m; i++) {
        const block_q4_0 * restrict x = (const block_q4_0 *)vx + (i * nb);

        sum_f = 0.0;

        for (int k = 0; k < nb; k++) {

            sum_i = 0;
            for (int kk = 0; kk < qk; kk++) {
                const int lo = (x[k].qs[kk] & 0x0F) - 8;
                const int hi = (x[k].qs[kk] >>   4) - 8;

                sum_i+= (lo * y[k].qs[kk]) + (hi * y[k].qs[kk + (qk / 2)]);
            }
            sum_f += sum_i * GGML_FP16_TO_FP32(x[k].d) * GGML_FP16_TO_FP32(y[k].d);
        }

        s[i] = sum_f;
    }
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
