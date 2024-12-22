#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <linux/ioctl.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_vec_dot_q4_0_q8_0_esp_riscv(int o, float *restrict s, size_t bs,
                                      const void *restrict vx, size_t bx,
                                      const void *restrict vy, size_t by,
                                      int mn);
void ggml_gemv_q4_0_4x4_q8_0_esp_riscv(int o, float *restrict s, size_t bs,
                             const void *restrict vx, const void *restrict vy,
                             int n, int m);
void ggml_gemm_q4_0_4x4_q8_0_esp_riscv(int o, float *restrict s, size_t bs,
                             const void *restrict vx, const void *restrict vy,
                             int n, int m);
void ggml_gemv_q4_0_q8_0_esp_riscv(int o, float *restrict s, size_t bs,
                                   const void *restrict vx, const void *restrict vy,
                                   int n, int m);
void ggml_gemm_q4_0_q8_0_esp_riscv(int o, float *restrict s, size_t bs,
                                   const void *restrict vx, const void *restrict vy,
                                   int n, int m);

bool esp_riscv_gemm(int64_t, int64_t, int64_t, const void *, int64_t,
                     const void *, int64_t, void *, int64_t, int, int,
                     int, int, int);

#ifdef __cplusplus
}
#endif
