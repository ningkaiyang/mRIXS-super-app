#include <stdlib.h>
#include <math.h>
#include <string.h>

void compute_masks_fast(
    const float* __restrict__ stddev,
    const float* __restrict__ residual,
    float s_th,
    float r_th,
    float* __restrict__ final_mask,
    float* __restrict__ stddev_mask,
    float* __restrict__ tail_mask,
    int n,
    int* __restrict__ out_surviving
) {
    int surviving = 0;
    for (int i = 0; i < n; i++) {
        float sv = stddev[i];
        float rv = residual[i];

        float ms = (isfinite(sv) && (sv < s_th)) ? 1.0f : 0.0f;
        float mr = (isfinite(rv) && (rv < r_th)) ? 1.0f : 0.0f;
        float mf = ms * mr;

        stddev_mask[i] = ms;
        tail_mask[i] = mr;
        final_mask[i] = mf;

        if (mf > 0.0f) {
            surviving++;
        }
    }
    *out_surviving = surviving;
}

static inline void swap_float(float* a, float* b) {
    float t = *a;
    *a = *b;
    *b = t;
}

static inline int partition_float(float arr[], int l, int r) {
    float x = arr[r];
    int i = l;
    for (int j = l; j <= r - 1; j++) {
        if (arr[j] <= x) {
            swap_float(&arr[i], &arr[j]);
            i++;
        }
    }
    swap_float(&arr[i], &arr[r]);
    return i;
}

static float quickselect_float(float arr[], int l, int r, int k) {
    while (l <= r) {
        if (l == r) return arr[l];
        int pivotIndex = l + (rand() % (r - l + 1));
        swap_float(&arr[pivotIndex], &arr[r]);
        int p = partition_float(arr, l, r);
        if (p == k) return arr[p];
        else if (p > k) r = p - 1;
        else l = p + 1;
    }
    return arr[l];
}

static float compute_median_exact(float buf[], int n_frames) {
    if (n_frames % 2 == 1) {
        return quickselect_float(buf, 0, n_frames - 1, n_frames / 2);
    } else {
        float m1 = quickselect_float(buf, 0, n_frames - 1, n_frames / 2 - 1);
        float m2 = quickselect_float(buf, n_frames / 2, n_frames - 1, n_frames / 2);
        return 0.5f * (m1 + m2);
    }
}

void compute_dark_stats_int16_c(
    const short* __restrict__ stack,
    int n_frames,
    int n_pixels,
    float q_ratio,
    float* __restrict__ out_med,
    float* __restrict__ out_std,
    float* __restrict__ out_pct93
) {
    float* buf = (float*)malloc(n_frames * sizeof(float));
    float* res = (float*)malloc(n_frames * sizeof(float));
    int k_pct = (int)roundf(q_ratio * (float)(n_frames - 1));

    for (int p = 0; p < n_pixels; p++) {
        double sum = 0.0;
        double sumsq = 0.0;
        for (int f = 0; f < n_frames; f++) {
            float val = (float)stack[f * n_pixels + p];
            buf[f] = val;
            sum += val;
            sumsq += (double)val * (double)val;
        }

        // Exact Median
        float med = compute_median_exact(buf, n_frames);
        out_med[p] = med;

        // StdDev
        double mean = sum / (double)n_frames;
        double var = (sumsq / (double)n_frames) - (mean * mean);
        out_std[p] = (var > 0.0) ? (float)sqrt(var) : 0.0f;

        // 93rd percentile of absolute residuals
        for (int f = 0; f < n_frames; f++) {
            float val = (float)stack[f * n_pixels + p];
            res[f] = fabsf(val - med);
        }
        out_pct93[p] = quickselect_float(res, 0, n_frames - 1, k_pct);
    }

    free(buf);
    free(res);
}

void compute_dark_stats_float_c(
    const float* __restrict__ stack,
    int n_frames,
    int n_pixels,
    float q_ratio,
    float* __restrict__ out_med,
    float* __restrict__ out_std,
    float* __restrict__ out_pct93
) {
    float* buf = (float*)malloc(n_frames * sizeof(float));
    float* res = (float*)malloc(n_frames * sizeof(float));
    int k_pct = (int)roundf(q_ratio * (float)(n_frames - 1));

    for (int p = 0; p < n_pixels; p++) {
        double sum = 0.0;
        double sumsq = 0.0;
        for (int f = 0; f < n_frames; f++) {
            float val = stack[f * n_pixels + p];
            buf[f] = val;
            sum += val;
            sumsq += (double)val * (double)val;
        }

        // Exact Median
        float med = compute_median_exact(buf, n_frames);
        out_med[p] = med;

        // StdDev
        double mean = sum / (double)n_frames;
        double var = (sumsq / (double)n_frames) - (mean * mean);
        out_std[p] = (var > 0.0) ? (float)sqrt(var) : 0.0f;

        // 93rd percentile of absolute residuals
        for (int f = 0; f < n_frames; f++) {
            float val = stack[f * n_pixels + p];
            res[f] = fabsf(val - med);
        }
        out_pct93[p] = quickselect_float(res, 0, n_frames - 1, k_pct);
    }

    free(buf);
    free(res);
}
