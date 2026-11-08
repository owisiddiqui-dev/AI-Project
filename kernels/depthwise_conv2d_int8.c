#include <stdint.h>
#include <riscv_vector.h>

/**
 * Optimized Int8 2D Depthwise Convolution Kernel for RISC-V Vector 1.0 (RVV 1.0)
 * Evaluated on CVA6 + Ara coprocessor.
 *
 * Parameters:
 *   out      : Output feature map tensor (C x H_out x W_out)
 *   in       : Input feature map tensor (C x H_in x W_in)
 *   weight   : Int8 weights (C x K x K)
 *   C        : Number of channels
 *   H_in     : Input height
 *   W_in     : Input width
 *   K        : Kernel size (3 for 3x3 depthwise)
 */
void depthwise_conv2d_3x3_int8(int8_t *out, const int8_t *in, const int8_t *weight,
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
                // Configure vector length for 8-bit elements, LMUL=2
                size_t vl = __riscv_vsetvl_e8m2(W_out - w);

                // Accumulator vector in 32-bit to prevent overflow
                vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

                // 3x3 unrolled convolution loop
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int8_t w_val = w_c[kh * 3 + kw];
                        const int8_t *in_ptr = in_c + (h + kh) * W_in + (w + kw);

                        // Vector load int8 activations
                        vint8m2_t vin = __riscv_vle8_v_i8m2(in_ptr, vl);

                        // Widening multiply & accumulate into 32-bit vector registers
                        vint16m4_t vprod = __riscv_vwmul_vx_i16m4(vin, w_val, vl);
                        vacc = __riscv_vwadd_wv_i32m8(vacc, vprod, vl);
                    }
                }

                // Saturate & narrow back down to int8 for output activations
                vint16m4_t vacc16 = __riscv_vnsra_wx_i16m4(vacc, 0, vl);
                vint8m2_t vres8 = __riscv_vnsra_wx_i8m2(vacc16, 0, vl);

                // Store output
                __riscv_vse8_v_i8m2(out_c + h * W_out + w, vres8, vl);

                w += vl;
            }
        }
    }
}
