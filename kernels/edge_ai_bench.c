#include "edge_ai_bench.h"
#include <riscv_vector.h>

/* =========================================================================
   1. Depthwise 3x3 Int8 Kernels
   ========================================================================= */

// NCHW layout (vectorized along spatial width W)
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
                        vint8m1_t vw = __riscv_vmv_v_x_i8m1(w_val, vl);
                        vint16m2_t vprod = __riscv_vwmul_vv_i16m2(vin, vw, vl);
                        vacc = __riscv_vwadd_wv_i32m4(vacc, vprod, vl);
                    }
                }

                vint32m4_t vclamped = __riscv_vmax_vx_i32m4(__riscv_vmin_vx_i32m4(vacc, 127, vl), -128, vl);
                vint16m2_t vacc16 = __riscv_vnsra_wx_i16m2(vclamped, 0, vl);
                vint8m1_t vres8 = __riscv_vnsra_wx_i8m1(vacc16, 0, vl);
                __riscv_vse8_v_i8m1(out_c + h * W_out + w, vres8, vl);
                w += vl;
            }
        }
    }
}

// NHWC layout (vectorized along channels C) - Standard in Edge AI
void depthwise_conv2d_3x3_int8_nhwc_m2(int8_t *out, const int8_t *in, const int8_t *weight,
                                      int64_t C, int64_t H_in, int64_t W_in) {
    int64_t H_out = H_in - 2;
    int64_t W_out = W_in - 2;

    for (int64_t h = 0; h < H_out; ++h) {
        for (int64_t w = 0; w < W_out; ++w) {
            int8_t *out_ptr = out + (h * W_out + w) * C;

            int64_t c = 0;
            while (c < C) {
                size_t vl = __riscv_vsetvl_e8m2(C - c);
                vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        const int8_t *in_ptr = in + ((h + kh) * W_in + (w + kw)) * C + c;
                        const int8_t *w_ptr = weight + (kh * 3 + kw) * C + c;

                        vint8m2_t vin = __riscv_vle8_v_i8m2(in_ptr, vl);
                        vint8m2_t vw = __riscv_vle8_v_i8m2(w_ptr, vl);

                        vint16m4_t vprod = __riscv_vwmul_vv_i16m4(vin, vw, vl);
                        vacc = __riscv_vwadd_wv_i32m8(vacc, vprod, vl);
                    }
                }

                vint32m8_t vclamped = __riscv_vmax_vx_i32m8(__riscv_vmin_vx_i32m8(vacc, 127, vl), -128, vl);
                vint16m4_t vacc16 = __riscv_vnsra_wx_i16m4(vclamped, 0, vl);
                vint8m2_t vres8 = __riscv_vnsra_wx_i8m2(vacc16, 0, vl);
                __riscv_vse8_v_i8m2(out_ptr + c, vres8, vl);
                c += vl;
            }
        }
    }
}

/* =========================================================================
   2. Pointwise 1x1 Int8 Kernel
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
                vint8m2_t vw = __riscv_vmv_v_x_i8m2(w_val, vl);
                vint16m4_t vprod = __riscv_vwmul_vv_i16m4(vin, vw, vl);
                vacc = __riscv_vwadd_wv_i32m8(vacc, vprod, vl);
            }

            vint32m8_t vclamped = __riscv_vmax_vx_i32m8(__riscv_vmin_vx_i32m8(vacc, 127, vl), -128, vl);
            vint16m4_t vacc16 = __riscv_vnsra_wx_i16m4(vclamped, 0, vl);
            vint8m2_t vres8 = __riscv_vnsra_wx_i8m2(vacc16, 0, vl);
            __riscv_vse8_v_i8m2(out_co + p, vres8, vl);
            p += vl;
        }
    }
}

/* =========================================================================
   3. Standard 2D Conv 3x3 Int8 (NHWC) - Output Channel Vectorized
   Weight layout: [3][3][Cin][Cout]
   ========================================================================= */

void conv2d_3x3_int8_nhwc_m2(int8_t *out, const int8_t *in, const int8_t *weight,
                            int64_t Cin, int64_t Cout, int64_t H_in, int64_t W_in) {
    int64_t H_out = H_in - 2;
    int64_t W_out = W_in - 2;

    for (int64_t h = 0; h < H_out; ++h) {
        for (int64_t w = 0; w < W_out; ++w) {
            int8_t *out_ptr = out + (h * W_out + w) * Cout;

            int64_t co = 0;
            while (co < Cout) {
                size_t vl = __riscv_vsetvl_e8m2(Cout - co);
                vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        for (int64_t ci = 0; ci < Cin; ++ci) {
                            int8_t in_val = in[((h + kh) * W_in + (w + kw)) * Cin + ci];
                            const int8_t *w_ptr = weight + (((kh * 3 + kw) * Cin + ci) * Cout) + co;

                            vint8m2_t vw = __riscv_vle8_v_i8m2(w_ptr, vl);
                            vint8m2_t vin = __riscv_vmv_v_x_i8m2(in_val, vl);
                            vint16m4_t vprod = __riscv_vwmul_vv_i16m4(vw, vin, vl);
                            vacc = __riscv_vwadd_wv_i32m8(vacc, vprod, vl);
                        }
                    }
                }

                vint32m8_t vclamped = __riscv_vmax_vx_i32m8(__riscv_vmin_vx_i32m8(vacc, 127, vl), -128, vl);
                vint16m4_t vacc16 = __riscv_vnsra_wx_i16m4(vclamped, 0, vl);
                vint8m2_t vres8 = __riscv_vnsra_wx_i8m2(vacc16, 0, vl);
                __riscv_vse8_v_i8m2(out_ptr + co, vres8, vl);

                co += vl;
            }
        }
    }
}

/* =========================================================================
   4. Residual Addition Int8 (Elementwise Saturating Add)
   ========================================================================= */

void add_residual_int8_m4(int8_t *out, const int8_t *in1, const int8_t *in2, size_t size) {
    size_t i = 0;
    while (i < size) {
        size_t vl = __riscv_vsetvl_e8m4(size - i);
        vint8m4_t va = __riscv_vle8_v_i8m4(in1 + i, vl);
        vint8m4_t vb = __riscv_vle8_v_i8m4(in2 + i, vl);
        vint8m4_t vres = __riscv_vsadd_vv_i8m4(va, vb, vl);
        __riscv_vse8_v_i8m4(out + i, vres, vl);
        i += vl;
    }
}

/* =========================================================================
   5. Global Average Pooling 2D Int8 (NHWC)
   ========================================================================= */

void global_avg_pool2d_int8_m2(int8_t *out, const int8_t *in, int64_t C, int64_t HW) {
    int64_t c = 0;
    while (c < C) {
        size_t vl = __riscv_vsetvl_e8m2(C - c);
        vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

        for (int64_t p = 0; p < HW; ++p) {
            vint8m2_t vin = __riscv_vle8_v_i8m2(in + p * C + c, vl);
            vint16m4_t vin16 = __riscv_vsext_vf2_i16m4(vin, vl);
            vacc = __riscv_vwadd_wv_i32m8(vacc, vin16, vl);
        }

        vint32m8_t vavg = __riscv_vdiv_vx_i32m8(vacc, (int32_t)HW, vl);
        vint32m8_t vclamped = __riscv_vmax_vx_i32m8(__riscv_vmin_vx_i32m8(vavg, 127, vl), -128, vl);
        vint16m4_t vavg16 = __riscv_vnsra_wx_i16m4(vclamped, 0, vl);
        vint8m2_t vres8 = __riscv_vnsra_wx_i8m2(vavg16, 0, vl);
        __riscv_vse8_v_i8m2(out + c, vres8, vl);

        c += vl;
    }
}

/* =========================================================================
   6. Fully Connected (GEMV) Int8
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

            vint16m8_t vprod = __riscv_vwmul_vv_i16m8(vin, vw, vl);
            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vsum = __riscv_vwredsum_vs_i16m8_i32m1(vprod, vzero, vl);
            acc += __riscv_vmv_x_s_i32m1_i32(vsum);

            i += vl;
        }
        if (acc > 127) acc = 127;
        if (acc < -128) acc = -128;
        out[o] = (int8_t)acc;
    }
}

/* =========================================================================
   7. Golden Reference Models (Scalar C)
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

void golden_depthwise_conv2d_3x3_int8_nhwc(int8_t *out, const int8_t *in, const int8_t *weight,
                                           int64_t C, int64_t H_in, int64_t W_in) {
    int64_t H_out = H_in - 2;
    int64_t W_out = W_in - 2;

    for (int64_t h = 0; h < H_out; ++h) {
        for (int64_t w = 0; w < W_out; ++w) {
            for (int64_t c = 0; c < C; ++c) {
                int32_t acc = 0;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int8_t in_val = in[((h + kh) * W_in + (w + kw)) * C + c];
                        int8_t w_val = weight[(kh * 3 + kw) * C + c];
                        acc += (int32_t)in_val * (int32_t)w_val;
                    }
                }
                if (acc > 127) acc = 127;
                if (acc < -128) acc = -128;
                out[(h * W_out + w) * C + c] = (int8_t)acc;
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

void golden_conv2d_3x3_int8_nhwc(int8_t *out, const int8_t *in, const int8_t *weight,
                                 int64_t Cin, int64_t Cout, int64_t H_in, int64_t W_in) {
    int64_t H_out = H_in - 2;
    int64_t W_out = W_in - 2;

    for (int64_t h = 0; h < H_out; ++h) {
        for (int64_t w = 0; w < W_out; ++w) {
            for (int64_t co = 0; co < Cout; ++co) {
                int32_t acc = 0;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        for (int64_t ci = 0; ci < Cin; ++ci) {
                            int8_t in_val = in[((h + kh) * W_in + (w + kw)) * Cin + ci];
                            int8_t w_val = weight[(((kh * 3 + kw) * Cin + ci) * Cout) + co];
                            acc += (int32_t)in_val * (int32_t)w_val;
                        }
                    }
                }
                if (acc > 127) acc = 127;
                if (acc < -128) acc = -128;
                out[(h * W_out + w) * Cout + co] = (int8_t)acc;
            }
        }
    }
}

void golden_add_residual_int8(int8_t *out, const int8_t *in1, const int8_t *in2, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        int32_t sum = (int32_t)in1[i] + (int32_t)in2[i];
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        out[i] = (int8_t)sum;
    }
}

void golden_global_avg_pool2d_int8(int8_t *out, const int8_t *in, int64_t C, int64_t HW) {
    for (int64_t c = 0; c < C; ++c) {
        int32_t sum = 0;
        for (int64_t p = 0; p < HW; ++p) {
            sum += (int32_t)in[p * C + c];
        }
        int32_t avg = sum / (int32_t)HW;
        if (avg > 127) avg = 127;
        if (avg < -128) avg = -128;
        out[c] = (int8_t)avg;
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
