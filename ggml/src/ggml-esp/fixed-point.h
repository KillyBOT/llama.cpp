#pragma once
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

static inline int32_t f32_to_i32(float value, uint32_t n_int_bits);
static inline uint32_t f32_to_u32(float value, uint32_t n_int_bits);
static inline int64_t f64_to_i64(double value, uint32_t n_int_bits);
static inline uint64_t f64_to_u64(double value, uint32_t n_int_bits);

static inline float i32_to_f32(int32_t value, uint32_t n_int_bits);
static inline float u32_to_f32(uint32_t value, uint32_t n_int_bits);
static inline double i64_to_f64(int64_t value, uint32_t n_int_bits);
static inline double u64_to_f64(uint64_t value, uint32_t n_int_bits);

/**
 * fixed_to_float - convert a fixed point value to floating point
 * @value: pointer to the value to be converted
 * @n_int_bits: number of integer bits, including sign bit
 *
 * Note: this assumes the fixed point value is 32 bits long
 */
static inline void i_to_f(void *value, uint32_t n_int_bits) {
  int32_t *valuep = (int32_t *)value;
  *(float *)value = i32_to_f32(*valuep, n_int_bits);
}

/**
 * float_to_fixed - convert a floating point value to fixed point
 * @value: pointer to the value to be converted
 * @n_int_bits: number of integer bits, including sign bit
 *
 * Note: this assumes the fixed point value is 32 bits long
 */
static inline void f_to_i(void *value, uint32_t n_int_bits) {
  float *p = (float *)value;
  *(uint32_t *)value = f32_to_i32(*p, n_int_bits);
}

static inline int32_t f32_to_i32(float value, uint32_t n_int_bits) {
  uint32_t shift_32 = 0x3f800000 + (0x800000 * (32 - n_int_bits));
  float *shift = (float *)&shift_32;

  return (int)(value * (*shift));
}

static inline uint32_t f32_to_u32(float value, uint32_t n_int_bits) {
  uint32_t shift_32 = 0x3f800000 + (0x800000 * (32 - n_int_bits));
  float *shift = (float *)(&shift_32);

  return (uint32_t)(value * (*shift));
}

static inline int64_t f64_to_i64(double value, uint32_t n_int_bits) {
  uint64_t shift_64 =
      0x3ff0000000000000LL + (0x0010000000000000LL * (64 - n_int_bits));
  double *shift = (double *)(&shift_64);

  return (int64_t)(value * (*shift));
}

static inline uint64_t f64_to_u64(double value, uint32_t n_int_bits) {
  uint64_t shift_64 =
      0x3ff0000000000000LL + (0x0010000000000000LL * (64 - n_int_bits));
  double *shift = (double *)(&shift_64);

  return (uint64_t)(value * (*shift));
}

static inline float i32_to_f32(int32_t value, uint32_t n_int_bits) {
  uint32_t shift_32 = 0x3f800000 - (0x800000 * (32 - n_int_bits));
  float *shift = (float *)(&shift_32);

  return (*shift) * (float)value;
}

static inline float u32_to_f32(uint32_t value, uint32_t n_int_bits) {
  uint32_t shift_32 = 0x3f800000 - (0x800000 * (32 - n_int_bits));
  float *shift = (float *)(&shift_32);

  return (*shift) * (float)value;
}

static inline double i64_to_f64(int64_t value, uint32_t n_int_bits) {
  uint64_t shift_64 =
      0x3ff0000000000000LL - (0x0010000000000000LL * (64 - n_int_bits));
  double *shift = (double *)(&shift_64);

  return (*shift) * (double)value;
}

static inline double u64_to_f64(uint64_t value, uint32_t n_int_bits) {
  uint64_t shift_ll =
      0x3ff0000000000000LL - (0x0010000000000000LL * (64 - n_int_bits));
  double *shift = (double *)(&shift_ll);

  return (*shift) * (double)value;
}

#ifdef __cplusplus
}
#endif
