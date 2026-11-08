#ifndef _EDGE_AI_BENCH_H_
#define _EDGE_AI_BENCH_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Depthwise Conv2D 3x3 Int8 (NCHW & NHWC)
void depthwise_conv2d_3x3_int8_m1(int8_t *out, const int8_t *in, const int8_t *weight,
                                 int64_t C, int64_t H_in, int64_t W_in);
void depthwise_conv2d_3x3_int8_nhwc_m2(int8_t *out, const int8_t *in, const int8_t *weight,
                                      int64_t C, int64_t H_in, int64_t W_in);

// Pointwise Conv2D 1x1 Int8
void pointwise_conv2d_1x1_int8_m2(int8_t *out, const int8_t *in, const int8_t *weight,
                                 int64_t C_in, int64_t C_out, int64_t HW);

// Standard Conv2D 3x3 Int8 (NHWC)
void conv2d_3x3_int8_nhwc_m2(int8_t *out, const int8_t *in, const int8_t *weight,
                            int64_t Cin, int64_t Cout, int64_t H_in, int64_t W_in);

// Residual Addition Int8 (Elementwise saturating addition)
void add_residual_int8_m4(int8_t *out, const int8_t *in1, const int8_t *in2, size_t size);

// Global Average Pooling 2D Int8 (NHWC)
void global_avg_pool2d_int8_m2(int8_t *out, const int8_t *in, int64_t C, int64_t HW);

// Fully Connected (GEMV) Int8
void fc_int8_m4(int8_t *out, const int8_t *in, const int8_t *weight,
                int64_t in_features, int64_t out_features);

// Golden Reference Models
void golden_depthwise_conv2d_3x3_int8(int8_t *out, const int8_t *in, const int8_t *weight,
                                      int64_t C, int64_t H_in, int64_t W_in);
void golden_depthwise_conv2d_3x3_int8_nhwc(int8_t *out, const int8_t *in, const int8_t *weight,
                                           int64_t C, int64_t H_in, int64_t W_in);
void golden_pointwise_conv2d_1x1_int8(int8_t *out, const int8_t *in, const int8_t *weight,
                                      int64_t C_in, int64_t C_out, int64_t HW);
void golden_conv2d_3x3_int8_nhwc(int8_t *out, const int8_t *in, const int8_t *weight,
                                 int64_t Cin, int64_t Cout, int64_t H_in, int64_t W_in);
void golden_add_residual_int8(int8_t *out, const int8_t *in1, const int8_t *in2, size_t size);
void golden_global_avg_pool2d_int8(int8_t *out, const int8_t *in, int64_t C, int64_t HW);
void golden_fc_int8(int8_t *out, const int8_t *in, const int8_t *weight,
                    int64_t in_features, int64_t out_features);

#ifdef __cplusplus
}
#endif

#endif // _EDGE_AI_BENCH_H_
