#include "edge_ai_bench.h"
#include <riscv_vector.h>

/* =========================================================================
   1. Depthwise 3x3 Int8 Kernels (LMUL = 1, 2, 4)
   ========================================================================= */

void depthwise_conv2d_3x3_int8_m1(int8_t *out, const int8_t *in, const int8_t *weight,
                                 int64_t C, int64_t H_in, int64_t W_in) {
    int64_t H_out = H_in - 2;
    int64_t W_out = W_in - 2;

    for (int64_t c = 0; c < C; ++c) {
        const int8_t *in_c = in + c * (H_in * W_in);
        int8_t *out_c = out + c * (H_out * W_out);
        const int8_t *w_c = weight + c * 9;

        for (int64_t h = 0; h < H_out; ++h) {
            int64_t w = 0;
            while (w < W_out) {
                size_t vl = __riscv_vsetvl_e8m1(W_out - w);
                vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int8_t w_val = w_c[kh * 3 + kw];
                        const int8_t *in_ptr = in_c + (h + kh) * W_in + (w + kw);
                        vint8m1_t vin = __riscv_vle8_v_i8m1(in_ptr, vl);
                        vint16m2_t vprod = __riscv_vwmul_vx_i16m2(vin, w_val, vl);
                        vacc = __riscv_vwadd_wv_i32m4(vacc, vprod, vl);
                    }
                }

                vint16m2_t vacc16 = __riscv_vnsra_wx_i16m2(vacc, 0, vl);
                vint8m1_t vres8 = __riscv_vnsra_wx_i8m1(vacc16, 0, vl);
                __riscv_vse8_v_i8m1(out_c + h * W_out + w, vres8, vl);
                w += vl;
            }
        }
    }
}

void depthwise_conv2d_3x3_int8_m2(int8_t *out, const int8_t *in, const int8_t *weight,
                                 int64_t C, int64_t H_in, int64_t W_in) {
    int64_t H_out = H_in - 2;
    int64_t W_out = W_in - 2;

    for (int64_t c = 0; c < C; ++c) {
        const int8_t *in_c = in + c * (H_in * W_in);
        int8_t *out_c = out + c * (H_out * W_out);
        const int8_t *w_c = weight + c * 9;

        for (int64_t h = 0; h < H_out; ++h) {
            int64_t w = 0;
            while (w < W_out) {
                size_t vl = __riscv_vsetvl_e8m2(W_out - w);
                vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int8_t w_val = w_c[kh * 3 + kw];
                        const int8_t *in_ptr = in_c + (h + kh) * W_in + (w + kw);
                        vint8m2_t vin = __riscv_vle8_v_i8m2(in_ptr, vl);
                        vint16m4_t vprod = __riscv_vwmul_vx_i16m4(vin, w_val, vl);
                        vacc = __riscv_vwadd_wv_i32m8(vacc, vprod, vl);
                    }
                }

                vint16m4_t vacc16 = __riscv_vnsra_wx_i16m4(vacc, 0, vl);
                vint8m2_t vres8 = __riscv_vnsra_wx_i8m2(vacc16, 0, vl);
                __riscv_vse8_v_i8m2(out_c + h * W_out + w, vres8, vl);
                w += vl;
            }
        }
    }
}

/* =========================================================================
   2. Pointwise 1x1 Int8 Kernels
   ========================================================================= */

void pointwise_conv2d_1x1_int8_m2(int8_t *out, const int8_t *in, const int8_t *weight,
                                 int64_t C_in, int64_t C_out, int64_t HW) {
    for (int64_t co = 0; co < C_out; ++co) {
        const int8_t *w_co = weight + co * C_in;
        int8_t *out_co = out + co * HW;

        int64_t p = 0;
        while (p < HW) {
            size_t vl = __riscv_vsetvl_e8m2(HW - p);
            vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

            for (int64_t ci = 0; ci < C_in; ++ci) {
                int8_t w_val = w_co[ci];
                const int8_t *in_ci = in + ci * HW + p;
                vint8m2_t vin = __riscv_vle8_v_i8m2(in_ci, vl);
                vint16m4_t vprod = __riscv_vwmul_vx_i16m4(vin, w_val, vl);
                vacc = __riscv_vwadd_wv_i32m8(vacc, vprod, vl);
            }

            vint16m4_t vacc16 = __riscv_vnsra_wx_i16m4(vacc, 0, vl);
            vint8m2_t vres8 = __riscv_vnsra_wx_i8m2(vacc16, 0, vl);
            __riscv_vse8_v_i8m2(out_co + p, vres8, vl);
            p += vl;
        }
    }
}

/* =========================================================================
   3. Fully Connected (GEMV) Int8 Kernel
   ========================================================================= */

void fc_int8_m4(int8_t *out, const int8_t *in, const int8_t *weight,
                int64_t in_features, int64_t out_features) {
    for (int64_t o = 0; o < out_features; ++o) {
        const int8_t *w_row = weight + o * in_features;

        int32_t acc = 0;
        int64_t i = 0;
        while (i < in_features) {
            size_t vl = __riscv_vsetvl_e8m4(in_features - i);
            vint8m4_t vin = __riscv_vle8_v_i8m4(in + i, vl);
            vint8m4_t vw = __riscv_vle8_v_i8m4(w_row + i, vl);

            // Vector widening multiply
            vint16m8_t vprod = __riscv_vwmul_vv_i16m8(vin, vw, vl);

            // Vector sum reduction into scalar
            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vsum = __riscv_vwredsum_vs_i16m8_i32m1(vprod, vzero, vl);
            acc += __riscv_vmv_x_s_i32m1_i32(vsum);

            i += vl;
        }
        // Clamping to int8 range [-128, 127]
        if (acc > 127) acc = 127;
        if (acc < -128) acc = -128;
        out[o] = (int8_t)acc;
    }
}

/* =========================================================================
   4. Golden Reference Models (Scalar C)
   ========================================================================= */

void golden_depthwise_conv2d_3x3_int8(int8_t *out, const int8_t *in, const int8_t *weight,
                                      int64_t C, int64_t H_in, int64_t W_in) {
    int64_t H_out = H_in - 2;
    int64_t W_out = W_in - 2;

    for (int64_t c = 0; c < C; ++c) {
        const int8_t *in_c = in + c * (H_in * W_in);
        int8_t *out_c = out + c * (H_out * W_out);
        const int8_t *w_c = weight + c * 9;

        for (int64_t h = 0; h < H_out; ++h) {
            for (int64_t w = 0; w < W_out; ++w) {
                int32_t acc = 0;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        acc += (int32_t)in_c[(h + kh) * W_in + (w + kw)] * (int32_t)w_c[kh * 3 + kw];
                    }
                }
                if (acc > 127) acc = 127;
                if (acc < -128) acc = -128;
                out_c[h * W_out + w] = (int8_t)acc;
            }
        }
    }
}

void golden_pointwise_conv2d_1x1_int8(int8_t *out, const int8_t *in, const int8_t *weight,
                                      int64_t C_in, int64_t C_out, int64_t HW) {
    for (int64_t co = 0; co < C_out; ++co) {
        for (int64_t p = 0; p < HW; ++p) {
            int32_t acc = 0;
            for (int64_t ci = 0; ci < C_in; ++ci) {
                acc += (int32_t)in[ci * HW + p] * (int32_t)weight[co * C_in + ci];
            }
            if (acc > 127) acc = 127;
            if (acc < -128) acc = -128;
            out[co * HW + p] = (int8_t)acc;
        }
    }
}

void golden_fc_int8(int8_t *out, const int8_t *in, const int8_t *weight,
                    int64_t in_features, int64_t out_features) {
    for (int64_t o = 0; o < out_features; ++o) {
        int32_t acc = 0;
        for (int64_t i = 0; i < in_features; ++i) {
            acc += (int32_t)in[i] * (int32_t)weight[o * in_features + i];
        }
        if (acc > 127) acc = 127;
        if (acc < -128) acc = -128;
        out[o] = (int8_t)acc;
    }
}
