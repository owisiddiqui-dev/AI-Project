#include <stdint.h>
#include <string.h>

#include "edge_ai_bench.h"
#include "runtime.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define MAX_CHANNELS 32
#define MAX_HW 300

// Buffers placed in L2 DRAM with Ara lane alignment
static int8_t in_buf[MAX_CHANNELS * MAX_HW] __attribute__((aligned(32 * NR_LANES), section(".l2")));
static int8_t in2_buf[MAX_CHANNELS * MAX_HW] __attribute__((aligned(32 * NR_LANES), section(".l2")));
static int8_t weight_buf[MAX_CHANNELS * MAX_CHANNELS * 9] __attribute__((aligned(32 * NR_LANES), section(".l2")));
static int8_t out_rvv[MAX_CHANNELS * MAX_HW] __attribute__((aligned(32 * NR_LANES), section(".l2")));
static int8_t out_gold[MAX_CHANNELS * MAX_HW] __attribute__((aligned(32 * NR_LANES), section(".l2")));

// Initialization with deterministic pseudo-random values
static void init_buffer(int8_t *buf, size_t size, int8_t seed) {
    for (size_t i = 0; i < size; ++i) {
        buf[i] = (int8_t)(((i * 7 + seed) % 255) - 128);
    }
}

// Verification helper
static int verify(const int8_t *a, const int8_t *b, size_t size, const char *name) {
    int errors = 0;
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            if (errors < 5) {
                printf("FAIL %s: idx=%d, rvv=%d, gold=%d\n", name, (int)i, (int)a[i], (int)b[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("PASS %s (verified %d elements)\n", name, (int)size);
        return 0;
    } else {
        printf("FAILED %s with %d mismatches!\n", name, errors);
        return 1;
    }
}

int main() {
    printf("====================================================================\n");
    printf(" Edge AI Quantized (Int8) Kernel Benchmark on CVA6 + Ara (RVV 1.0) \n");
    printf(" Hardware: NR_LANES=%d, VLEN=%d bits\n", NR_LANES, VLEN);
    printf(" Workloads: MLPerf Tiny DS-CNN & ResNet-8 Core Operator Suite     \n");
    printf("====================================================================\n\n");

    int total_failures = 0;
    int64_t t_start, t_end;
    int64_t cycles_scalar, cycles_rvv;

    /* ---------------------------------------------------------------------
       Benchmark 1A: KWS DS-CNN Depthwise 3x3 Int8 (NCHW Layout)
       --------------------------------------------------------------------- */
    int64_t C = 16, H_in = 25, W_in = 10;
    int64_t H_out = H_in - 2;
    int64_t W_out = W_in - 2;
    size_t in_size = C * H_in * W_in;
    size_t out_size = C * H_out * W_out;
    size_t weight_size = C * 9;

    init_buffer(in_buf, in_size, 3);
    init_buffer(weight_buf, weight_size, 5);

    printf("--- [Benchmark 1A] Depthwise Conv2D 3x3 NCHW (C=%d, H=%d, W=%d) ---\n", (int)C, (int)H_in, (int)W_in);
    t_start = get_cycle_count();
    golden_depthwise_conv2d_3x3_int8(out_gold, in_buf, weight_buf, C, H_in, W_in);
    t_end = get_cycle_count();
    cycles_scalar = t_end - t_start;

    memset(out_rvv, 0, out_size);
    t_start = get_cycle_count();
    depthwise_conv2d_3x3_int8_m1(out_rvv, in_buf, weight_buf, C, H_in, W_in);
    t_end = get_cycle_count();
    cycles_rvv = t_end - t_start;
    total_failures += verify(out_rvv, out_gold, out_size, "Depthwise_3x3_NCHW_m1");
    printf("  Cycles Scalar: %ld, Cycles RVV: %ld (Speedup: %d.%02dx)\n\n",
           (long)cycles_scalar, (long)cycles_rvv,
           (int)(cycles_scalar / cycles_rvv), (int)((cycles_scalar * 100 / cycles_rvv) % 100));

    /* ---------------------------------------------------------------------
       Benchmark 1B: KWS DS-CNN Depthwise 3x3 Int8 (NHWC Layout - Channel-Tiled)
       --------------------------------------------------------------------- */
    printf("--- [Benchmark 1B] Depthwise Conv2D 3x3 NHWC (C=%d, H=%d, W=%d) ---\n", (int)C, (int)H_in, (int)W_in);
    init_buffer(in_buf, in_size, 3);
    init_buffer(weight_buf, weight_size, 5);

    t_start = get_cycle_count();
    golden_depthwise_conv2d_3x3_int8_nhwc(out_gold, in_buf, weight_buf, C, H_in, W_in);
    t_end = get_cycle_count();
    cycles_scalar = t_end - t_start;

    memset(out_rvv, 0, out_size);
    t_start = get_cycle_count();
    depthwise_conv2d_3x3_int8_nhwc_m2(out_rvv, in_buf, weight_buf, C, H_in, W_in);
    t_end = get_cycle_count();
    cycles_rvv = t_end - t_start;
    total_failures += verify(out_rvv, out_gold, out_size, "Depthwise_3x3_NHWC_m2");
    printf("  Cycles Scalar: %ld, Cycles RVV: %ld (Speedup: %d.%02dx)\n\n",
           (long)cycles_scalar, (long)cycles_rvv,
           (int)(cycles_scalar / cycles_rvv), (int)((cycles_scalar * 100 / cycles_rvv) % 100));

    /* ---------------------------------------------------------------------
       Benchmark 2: Pointwise Conv2D 1x1 Int8 (Cin=16, Cout=16, HW=184)
       --------------------------------------------------------------------- */
    int64_t Cin = 16, Cout = 16, HW = 184;
    size_t pw_in_size = Cin * HW;
    size_t pw_out_size = Cout * HW;
    size_t pw_w_size = Cout * Cin;

    init_buffer(in_buf, pw_in_size, 7);
    init_buffer(weight_buf, pw_w_size, 9);

    printf("--- [Benchmark 2] Pointwise Conv2D 1x1 Int8 (Cin=%d, Cout=%d, HW=%d) ---\n", (int)Cin, (int)Cout, (int)HW);
    t_start = get_cycle_count();
    golden_pointwise_conv2d_1x1_int8(out_gold, in_buf, weight_buf, Cin, Cout, HW);
    t_end = get_cycle_count();
    cycles_scalar = t_end - t_start;

    memset(out_rvv, 0, pw_out_size);
    t_start = get_cycle_count();
    pointwise_conv2d_1x1_int8_m2(out_rvv, in_buf, weight_buf, Cin, Cout, HW);
    t_end = get_cycle_count();
    cycles_rvv = t_end - t_start;
    total_failures += verify(out_rvv, out_gold, pw_out_size, "Pointwise_1x1_LMUL2");
    printf("  Cycles Scalar: %ld, Cycles RVV: %ld (Speedup: %d.%02dx)\n\n",
           (long)cycles_scalar, (long)cycles_rvv,
           (int)(cycles_scalar / cycles_rvv), (int)((cycles_scalar * 100 / cycles_rvv) % 100));

    /* ---------------------------------------------------------------------
       Benchmark 3: ResNet-8 Standard Conv2D 3x3 NHWC (Cin=8, Cout=8, H=10, W=10)
       --------------------------------------------------------------------- */
    int64_t conv_cin = 8, conv_cout = 8, conv_h = 10, conv_w = 10;
    int64_t conv_h_out = conv_h - 2, conv_w_out = conv_w - 2;
    size_t conv_in_size = conv_h * conv_w * conv_cin;
    size_t conv_out_size = conv_h_out * conv_w_out * conv_cout;
    size_t conv_w_size = conv_cout * 9 * conv_cin;

    init_buffer(in_buf, conv_in_size, 11);
    init_buffer(weight_buf, conv_w_size, 13);

    printf("--- [Benchmark 3] ResNet-8 Conv2D 3x3 NHWC (Cin=%d, Cout=%d, H=%d, W=%d) ---\n",
           (int)conv_cin, (int)conv_cout, (int)conv_h, (int)conv_w);
    t_start = get_cycle_count();
    golden_conv2d_3x3_int8_nhwc(out_gold, in_buf, weight_buf, conv_cin, conv_cout, conv_h, conv_w);
    t_end = get_cycle_count();
    cycles_scalar = t_end - t_start;

    memset(out_rvv, 0, conv_out_size);
    t_start = get_cycle_count();
    conv2d_3x3_int8_nhwc_m2(out_rvv, in_buf, weight_buf, conv_cin, conv_cout, conv_h, conv_w);
    t_end = get_cycle_count();
    cycles_rvv = t_end - t_start;
    total_failures += verify(out_rvv, out_gold, conv_out_size, "Conv2D_3x3_NHWC_m2");
    printf("  Cycles Scalar: %ld, Cycles RVV: %ld (Speedup: %d.%02dx)\n\n",
           (long)cycles_scalar, (long)cycles_rvv,
           (int)(cycles_scalar / cycles_rvv), (int)((cycles_scalar * 100 / cycles_rvv) % 100));

    /* ---------------------------------------------------------------------
       Benchmark 4: Residual Addition (Skip Connection) Int8 (Size=512)
       --------------------------------------------------------------------- */
    size_t res_size = 512;
    init_buffer(in_buf, res_size, 17);
    init_buffer(in2_buf, res_size, 19);

    printf("--- [Benchmark 4] ResNet-8 Residual Addition Int8 (Size=%d) ---\n", (int)res_size);
    t_start = get_cycle_count();
    golden_add_residual_int8(out_gold, in_buf, in2_buf, res_size);
    t_end = get_cycle_count();
    cycles_scalar = t_end - t_start;

    memset(out_rvv, 0, res_size);
    t_start = get_cycle_count();
    add_residual_int8_m4(out_rvv, in_buf, in2_buf, res_size);
    t_end = get_cycle_count();
    cycles_rvv = t_end - t_start;
    total_failures += verify(out_rvv, out_gold, res_size, "Residual_Add_LMUL4");
    printf("  Cycles Scalar: %ld, Cycles RVV: %ld (Speedup: %d.%02dx)\n\n",
           (long)cycles_scalar, (long)cycles_rvv,
           (int)(cycles_scalar / cycles_rvv), (int)((cycles_scalar * 100 / cycles_rvv) % 100));

    /* ---------------------------------------------------------------------
       Benchmark 5: Global Average Pooling Int8 (C=16, HW=64)
       --------------------------------------------------------------------- */
    int64_t pool_c = 16, pool_hw = 64;
    size_t pool_in_size = pool_c * pool_hw;
    init_buffer(in_buf, pool_in_size, 23);

    printf("--- [Benchmark 5] Global Average Pooling Int8 (C=%d, HW=%d) ---\n", (int)pool_c, (int)pool_hw);
    t_start = get_cycle_count();
    golden_global_avg_pool2d_int8(out_gold, in_buf, pool_c, pool_hw);
    t_end = get_cycle_count();
    cycles_scalar = t_end - t_start;

    memset(out_rvv, 0, pool_c);
    t_start = get_cycle_count();
    global_avg_pool2d_int8_m2(out_rvv, in_buf, pool_c, pool_hw);
    t_end = get_cycle_count();
    cycles_rvv = t_end - t_start;
    total_failures += verify(out_rvv, out_gold, pool_c, "Global_AvgPool_LMUL2");
    printf("  Cycles Scalar: %ld, Cycles RVV: %ld (Speedup: %d.%02dx)\n\n",
           (long)cycles_scalar, (long)cycles_rvv,
           (int)(cycles_scalar / cycles_rvv), (int)((cycles_scalar * 100 / cycles_rvv) % 100));

    /* ---------------------------------------------------------------------
       Benchmark 6: Fully Connected Classifier (In=64, Out=12)
       --------------------------------------------------------------------- */
    int64_t in_features = 64, out_features = 12;
    init_buffer(in_buf, in_features, 29);
    init_buffer(weight_buf, in_features * out_features, 31);

    printf("--- [Benchmark 6] Fully Connected Classifier Int8 (In=%d, Out=%d) ---\n", (int)in_features, (int)out_features);
    t_start = get_cycle_count();
    golden_fc_int8(out_gold, in_buf, weight_buf, in_features, out_features);
    t_end = get_cycle_count();
    cycles_scalar = t_end - t_start;

    memset(out_rvv, 0, out_features);
    t_start = get_cycle_count();
    fc_int8_m4(out_rvv, in_buf, weight_buf, in_features, out_features);
    t_end = get_cycle_count();
    cycles_rvv = t_end - t_start;
    total_failures += verify(out_rvv, out_gold, out_features, "FC_Classifier_LMUL4");
    printf("  Cycles Scalar: %ld, Cycles RVV: %ld (Speedup: %d.%02dx)\n\n",
           (long)cycles_scalar, (long)cycles_rvv,
           (int)(cycles_scalar / cycles_rvv), (int)((cycles_scalar * 100 / cycles_rvv) % 100));

    printf("====================================================================\n");
    if (total_failures == 0) {
        printf(" ALL BENCHMARKS PASSED SUCCESSFULLY!\n");
    } else {
        printf(" BENCHMARK COMPLETED WITH %d FAILURES.\n", total_failures);
    }
    printf("====================================================================\n");

    return total_failures;
}
