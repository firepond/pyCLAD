/*
   Copyright 2021 FogML
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "fogml_lof.h"

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#if !defined(TINYML_DISABLE_NEON)
#if defined(__ARM_NEON) && defined(__aarch64__)
#define TINYML_USE_NEON 1
#include <arm_neon.h>
#endif
#endif

#define LOF_VECTOR(i, config) &(config->data[i * config->vector_size])
#define TINYML_MAX_DISTANCE FLT_MAX

void tinyml_lof_init(tinyml_lof_config_t *config) {
}

// Fast squared Euclidean distance (no pow, no sqrt). Vectorized.
static inline float tinyml_lof_sq_distance_vec(const float *restrict a, const float *restrict b, int len) {
#if defined(TINYML_USE_NEON)
    // AArch64 NEON path (Apple Silicon)
    float32x4_t acc4 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i + 0);
        float32x4_t vb0 = vld1q_f32(b + i + 0);
        float32x4_t vd0 = vsubq_f32(va0, vb0);
        acc4 = vmlaq_f32(acc4, vd0, vd0);  // acc += d*d

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t vd1 = vsubq_f32(va1, vb1);
        acc4 = vmlaq_f32(acc4, vd1, vd1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t vd2 = vsubq_f32(va2, vb2);
        acc4 = vmlaq_f32(acc4, vd2, vd2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        float32x4_t vd3 = vsubq_f32(va3, vb3);
        acc4 = vmlaq_f32(acc4, vd3, vd3);
    }
    float acc = vaddvq_f32(acc4);
    for (; i + 4 <= len; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vd = vsubq_f32(va, vb);
        float32x4_t sq = vmulq_f32(vd, vd);
        acc += vaddvq_f32(sq);
    }
    for (; i < len; ++i) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
#else
    // Portable path: simple form that auto-vectorizes well
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int i = 0;
    int const len_minus_16 = len - 16;
    // #pragma omp simd reduction(+ : acc0, acc1, acc2, acc3)
    for (i = 0; i <= len_minus_16; i += 16) {
        float d0 = a[i + 0] - b[i + 0];
        acc0 += d0 * d0;
        float d1 = a[i + 1] - b[i + 1];
        acc1 += d1 * d1;
        float d2 = a[i + 2] - b[i + 2];
        acc2 += d2 * d2;
        float d3 = a[i + 3] - b[i + 3];
        acc3 += d3 * d3;
        float d4 = a[i + 4] - b[i + 4];
        acc0 += d4 * d4;
        float d5 = a[i + 5] - b[i + 5];
        acc1 += d5 * d5;
        float d6 = a[i + 6] - b[i + 6];
        acc2 += d6 * d6;
        float d7 = a[i + 7] - b[i + 7];
        acc3 += d7 * d7;
        float d8 = a[i + 8] - b[i + 8];
        acc0 += d8 * d8;
        float d9 = a[i + 9] - b[i + 9];
        acc1 += d9 * d9;
        float dA = a[i + 10] - b[i + 10];
        acc2 += dA * dA;
        float dB = a[i + 11] - b[i + 11];
        acc3 += dB * dB;
        float dC = a[i + 12] - b[i + 12];
        acc0 += dC * dC;
        float dD = a[i + 13] - b[i + 13];
        acc1 += dD * dD;
        float dE = a[i + 14] - b[i + 14];
        acc2 += dE * dE;
        float dF = a[i + 15] - b[i + 15];
        acc3 += dF * dF;
    }
    float acc = acc0 + acc1 + acc2 + acc3;
    int t = i;
    for (i = t; i < len; ++i) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
#endif
}

// Optional: keep the original function, but make it use the fast inner loop
float tinyml_lof_normal_distance_vec(float *vec_a, float *vec_b, int len) {
    // Compute sqrt only when a true distance is required.
    return sqrtf(tinyml_lof_sq_distance_vec(vec_a, vec_b, len));
}

// Single-pass top-k selection (keeps a sorted array of size k). O(N * k) but only one distance per pair.
// neigh_d2 keeps squared distances ascending; neigh_idx keeps indices aligned to neigh_d2.
static inline void tinyml_lof_k_neighbours_topk(int a, int *neigh_idx, float *neigh_d2, tinyml_lof_config_t *config) {
    const int k = config->parameter_k;
    int size = 0;

    const float *base = LOF_VECTOR(a, config);
    for (int i = 0; i < config->n; i++) {
        if (i == a)
            continue;

        float d2 = tinyml_lof_sq_distance_vec(base, LOF_VECTOR(i, config), config->vector_size);

        if (size < k) {
            // Insert in sorted order
            int pos = size;
            while (pos > 0 && d2 < neigh_d2[pos - 1]) {
                neigh_d2[pos] = neigh_d2[pos - 1];
                neigh_idx[pos] = neigh_idx[pos - 1];
                pos--;
            }
            neigh_d2[pos] = d2;
            neigh_idx[pos] = i;
            size++;
        } else if (d2 < neigh_d2[k - 1]) {
            // Replace worst and keep sorted
            int pos = k - 1;
            while (pos > 0 && d2 < neigh_d2[pos - 1]) {
                neigh_d2[pos] = neigh_d2[pos - 1];
                neigh_idx[pos] = neigh_idx[pos - 1];
                pos--;
            }
            neigh_d2[pos] = d2;
            neigh_idx[pos] = i;
        }
    }
}

// Top-k neighbors for an external query vector against the training set.
// Returns indices and squared distances (ascending).
static inline void tinyml_lof_k_neighbours_topk_query(const float *restrict query,
                                                      int *restrict neigh_idx,
                                                      float *restrict neigh_d2,
                                                      const tinyml_lof_config_t *restrict config) {
    const int k = config->parameter_k;
    int size = 0;

    for (int i = 0; i < config->n; i++) {
        // Prefetch next row to hide memory latency for higher D
        if (i + 1 < config->n) {
            __builtin_prefetch(LOF_VECTOR(i + 1, config), 0, 1);
        }

        const float *base = LOF_VECTOR(i, config);
        float d2 = tinyml_lof_sq_distance_vec(query, base, config->vector_size);

        if (size < k) {
            int pos = size;
            while (pos > 0 && d2 < neigh_d2[pos - 1]) {
                neigh_d2[pos] = neigh_d2[pos - 1];
                neigh_idx[pos] = neigh_idx[pos - 1];
                pos--;
            }
            neigh_d2[pos] = d2;
            neigh_idx[pos] = i;
            size++;
        } else if (d2 < neigh_d2[k - 1]) {
            int pos = k - 1;
            while (pos > 0 && d2 < neigh_d2[pos - 1]) {
                neigh_d2[pos] = neigh_d2[pos - 1];
                neigh_idx[pos] = neigh_idx[pos - 1];
                pos--;
            }
            neigh_d2[pos] = d2;
            neigh_idx[pos] = i;
        }
    }
}

void tinyml_lof_k_neighbours_vec(float *vector, int *neighbours, tinyml_lof_config_t *config) {
    for (int k = 0; k < config->parameter_k; k++) {
        float max = TINYML_MAX_DISTANCE;

        for (int i = 0; i < config->n; i++) {
            bool used = false;
            for (int j = 0; j < k; j++) {
                if (neighbours[j] == i) {
                    used = true;
                    break;
                }
            }
            if (used)
                continue;

            float dist = tinyml_lof_normal_distance_vec(vector, LOF_VECTOR(i, config), config->vector_size);
            if (dist < max) {
                max = dist;
                neighbours[k] = i;
            };
        }
    }
}

void tinyml_lof_k_neighbours(int a, int *neighbours, tinyml_lof_config_t *config) {
    for (int k = 0; k < config->parameter_k; k++) {
        float max = TINYML_MAX_DISTANCE;

        for (int i = 0; i < config->n; i++) {
            if (a == i)
                continue;  // the same node

            bool used = false;
            for (int j = 0; j < k; j++) {
                if (neighbours[j] == i) {
                    used = true;
                    break;
                }
            }
            if (used)
                continue;

            float dist = tinyml_lof_normal_distance_vec(LOF_VECTOR(a, config), LOF_VECTOR(i, config), config->vector_size);
            if (dist < max) {
                max = dist;
                neighbours[k] = i;
            };
        }
    }
}

float tinyml_max(float a, float b) {
    return a > b ? a : b;
}

float tinyml_lof_reachability_distance(float *vector, int b, tinyml_lof_config_t *config) {
    return tinyml_max(config->k_distance[b], tinyml_lof_normal_distance_vec(vector, LOF_VECTOR(b, config), config->vector_size));
}

float tinyml_lof_reachability_density(float *vector, int *neighbours, tinyml_lof_config_t *config) {
    float lrd = 0;

    for (int k = 0; k < config->parameter_k; k++) {
        lrd += tinyml_lof_reachability_distance(vector, neighbours[k], config);
    }

    lrd = lrd / config->parameter_k;
    lrd = 1 / lrd;

    return lrd;
}

float tinyml_lof_score(float *vector, tinyml_lof_config_t *config) {
    // measure running time
    clock_t start = clock();

    int neighbours[10];

    float score = 0;

    tinyml_lof_k_neighbours_vec(vector, neighbours, config);
    // tinyml_lof_k_neighbours(a, neighbours, config);

    for (int k = 0; k < config->parameter_k; k++) {
        score += config->lrd[neighbours[k]];
    }

    score /= tinyml_lof_reachability_density(vector, neighbours, config);

    score /= config->parameter_k;

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("Time taken: %f seconds\n", cpu_time_used);
    return score;
}

void tinyml_lof_score_vectored(float **vector, tinyml_lof_config_t *config, float *scores, int size) {
    const int k = config->parameter_k;
    const float *restrict kdist = config->k_distance;
    const float *restrict lrd = config->lrd;

    // Use threads only when batch is large enough
    // #pragma omp parallel for if (size >= 64) schedule(static)
    for (int qi = 0; qi < size; qi++) {
        int neigh_idx[k];
        float neigh_d2[k];
        tinyml_lof_k_neighbours_topk_query(vector[qi], neigh_idx, neigh_d2, config);

        float sum_lrd = 0.0f;
        for (int j = 0; j < k; j++) sum_lrd += lrd[neigh_idx[j]];
        float mean_lrd_neighbors = sum_lrd / (float)k;

        float sum_reach = 0.0f;
        for (int j = 0; j < k; j++) {
            int b = neigh_idx[j];
            float dist = sqrtf(neigh_d2[j]);
            float reach = fmaxf(kdist[b], dist);
            sum_reach += reach;
        }
        scores[qi] = mean_lrd_neighbors / (1.0f / (sum_reach / (float)k));
    }
}

// New: contiguous matrix scorer to avoid float** overhead from Python
void tinyml_lof_score_matrix(const float *restrict vectors,
                             tinyml_lof_config_t *restrict config,
                             float *restrict scores,
                             int size) {
    const int d = config->vector_size;
    const int k = config->parameter_k;
    const float *restrict kdist = config->k_distance;
    const float *restrict lrd = config->lrd;
    // printf("data size:%d\n", size);

    // #pragma omp parallel for schedule(static)
    for (int qi = 0; qi < size; qi++) {
        const float *query = &vectors[(size_t)qi * d];

        int neigh_idx[k];
        float neigh_d2[k];
        tinyml_lof_k_neighbours_topk_query(query, neigh_idx, neigh_d2, config);

        float sum_lrd = 0.0f;
        for (int j = 0; j < k; j++) sum_lrd += lrd[neigh_idx[j]];
        float mean_lrd_neighbors = sum_lrd / (float)k;

        float sum_reach = 0.0f;
        for (int j = 0; j < k; j++) {
            int b = neigh_idx[j];
            float dist = sqrtf(neigh_d2[j]);
            float reach = fmaxf(kdist[b], dist);
            sum_reach += reach;
        }
        float lrd_query = 1.0f / (sum_reach / (float)k);
        scores[qi] = mean_lrd_neighbors / lrd_query;
    }
}

void tinyml_lof_learn(tinyml_lof_config_t *config) {
    // measure CPU time
    clock_t start = clock();

    const int n = config->n;
    const int k = config->parameter_k;
    if (n < (k + 1))
        return;

    // Allocate neighbor caches: n x k (indices and squared distances)
    int *all_neigh = (int *)malloc((size_t)n * k * sizeof(int));
    float *all_neigh_d2 = (float *)malloc((size_t)n * k * sizeof(float));
    if (!all_neigh || !all_neigh_d2) {
        free(all_neigh);
        free(all_neigh_d2);
        return;
    }

    // 1) Compute neighbors once and k-distance for every point
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int *neigh_idx = &all_neigh[i * k];
        float *neigh_d2 = &all_neigh_d2[i * k];
        tinyml_lof_k_neighbours_topk(i, neigh_idx, neigh_d2, config);

        // k-distance is sqrt of the k-th smallest squared distance
        config->k_distance[i] = sqrtf(neigh_d2[k - 1]);
    }

    // 2) Compute lrd using cached neighbors and their d2 (no extra distance recomputation)
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int *neigh_idx = &all_neigh[i * k];
        float *neigh_d2 = &all_neigh_d2[i * k];

        float sum_reach = 0.0f;
        for (int j = 0; j < k; j++) {
            int b = neigh_idx[j];
            // reachability_distance = max(k_distance[b], distance(i,b))
            float dist_ib = sqrtf(neigh_d2[j]);                   // reuse squared dist
            float reach = fmaxf(config->k_distance[b], dist_ib);  // both are true distances
            sum_reach += reach;
        }
        float lrd = sum_reach / (float)k;
        config->lrd[i] = 1.0f / lrd;
    }

    free(all_neigh);
    free(all_neigh_d2);

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC / 10;
    // printf("Learn Time taken: %f seconds\n", cpu_time_used);
}
