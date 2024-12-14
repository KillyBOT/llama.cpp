#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <linux/ioctl.h>

#include "ggml-common.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_vec_dot_q4_0_q8_0_esp(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

bool esp_riscv_gemm(int64_t, int64_t, int64_t, const void *, int64_t,
                     const void *, int64_t, void *, int64_t, int, int,
                     int, int, int);

#ifdef __cplusplus
}
#endif
