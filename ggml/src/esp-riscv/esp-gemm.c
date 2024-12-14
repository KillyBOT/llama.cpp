#include "esp-riscv/esp.h"
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#include "esp-acc-prints.h"
#include "esp-gemm-cfg.h"
#include "esp-gemm.h"

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
bool esp_riscv_gemm(int64_t m, int64_t n, int64_t k, const void *A, int64_t lda,
                    const void *B, int64_t ldb, void *C, int64_t ldc, int ith,
                    int nth, int Atype, int Btype, int Ctype) {

    size_t src0_len, src1_len, dest_len, src0_size, src1_size, dest_size;
    float *accel_buf;
    float *aqk, *bqk, *cqk;

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(nth > 0); // Unused for now
    assert(ith < nth); // Unused for now

    if (n < 2)
        return false;

    if (Ctype != GGML_TYPE_F32)
        return false;

    switch (Atype) {
    case GGML_TYPE_Q4_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;

        src0_len = m * k;
        src1_len = k * n;
        dest_len = m * n;
        src0_size = src0_len * sizeof(float);
        src1_size = src1_len * sizeof(float);
        dest_size = dest_len * sizeof(float);

        /* Allocate a buffer for the accelerator. For now, just make it max size */
        // accel_buf = (token_t *)esp_alloc(MAX_SIZE);
        // cfg_000[0].hw_buf = accel_buf;

        gemm_cfg_000[0].do_relu = 0;
        gemm_cfg_000[0].transpose = 1;
        gemm_cfg_000[0].ninputs = 1; // TODO: Batch more inputs together?
        gemm_cfg_000[0].d1 = m;
        gemm_cfg_000[0].d2 = k;
        gemm_cfg_000[0].d3 = n;

        print_gemm_cfg(cfg_000, gemm_cfg_000);

        accel_buf = (float *)esp_alloc(MAX_SIZE);

        aqk = accel_buf;
        bqk = accel_buf + src0_len;

        // int64_t ytiles = (m - m0) / RM;
        // int64_t xtiles = (n - n0) / RN;
        // int64_t ntiles = xtiles * ytiles;
        // int64_t duty = (tiles + nth - 1) / nth;
        // int64_t start = duty * ith;
        // int64_t end = start + duty;
        // if (end > tiles)
        //     end = tiles;
        // for (int64_t job = start; job < end; ++job) {
        //     int64_t ii = m0 + job / xtiles * RM;
        //     int64_t jj = n0 + job % xtiles * RN;
        //     D Cv[RN][RM] = {};
        //     for (int64_t l = 0; l < k; l += KN)
        //         for (int64_t j = 0; j < RN; ++j)
        //             for (int64_t i = 0; i < RM; ++i)
        //                 Cv[j][i] = madd(load<V>(A + lda * (ii + i) + l),
        //                                 load<V>(B + ldb * (jj + j) + l),
        //                                 Cv[j][i]);
        //     for (int64_t j = 0; j < RN; ++j)
        //         for (int64_t i = 0; i < RM; ++i)
        //             C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        // }
        //
        esp_free(accel_buf);

        return false;
    }

    default:
        return false;
    }

    /* Stupid trick so that the compiler doesn't complain that these values are
    * never used */
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)ith;
    (void)nth;
    (void)Atype;
    (void)Btype;
    (void)Ctype;

}

void ggml_vec_dot_q4_0_q8_0_esp(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK4_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;
    float *accel_buf;

    // int ib = 0;
    // float sumf = 0;
    //
    // float xq[QK4_0];
    // float yq[QK8_0];
    //
    // for (; ib < nb; ib++) {
    //     dequantize_row_q4_0(x + ib, xq, QK4_0);
    //     dequantize_row_q8_0(y + ib, yq, QK4_0);
    //     for (int i = 0; i < qk; i++) {
    //         sumf += xq[i] * yq[i];
    //     }
    // }
    //
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
    gemm_cfg_000[0].d2 = n;
    gemm_cfg_000[0].d3 = 1;
    gemm_cfg_000[0].ld_offset1 = 0;
    gemm_cfg_000[0].ld_offset2 = n;
    gemm_cfg_000[0].st_offset = 2 * n;

    print_gemm_cfg(cfg_000, gemm_cfg_000);

    accel_buf = (float *)esp_alloc(MAX_SIZE);

    dequantize_row_q4_0(x, accel_buf, n);
    dequantize_row_q8_0(y, accel_buf + n, n);

    cfg_000[0].hw_buf = accel_buf;

    esp_run(cfg_000, NACC);

    *s = accel_buf[2 * n];

    // TODO: Free the buffer at the end of execution, not every time this function is called (probably via C++)
    esp_free(accel_buf);
}
