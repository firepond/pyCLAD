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

#define LOF_VECTOR(i, config) &(config->data[i * config->vector_size])
#define TINYML_MAX_DISTANCE FLT_MAX

void tinyml_lof_init(tinyml_lof_config_t *config) {
}

// Fast squared Euclidean distance (no pow, no sqrt). Vectorized.
static inline float tinyml_lof_sq_distance_vec(const float *restrict a, const float *restrict b, int len) {
    float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
    for (int i = 0; i < len; i++) {
        float d = a[i] - b[i];
        acc = fmaf(d, d, acc);  // acc += d*d
    }
    return acc;
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
                }  // was used before
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
                }  // was used before
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
    // measure running time
    clock_t start = clock();

    const int k = config->parameter_k;
    const float *restrict kdist = config->k_distance;
    const float *restrict lrd = config->lrd;

#pragma omp parallel for schedule(static)
    for (int qi = 0; qi < size; qi++) {
        // Per-query top-k (on stack, no malloc). Requires C99 VLA.
        int neigh_idx[k];
        float neigh_d2[k];

        tinyml_lof_k_neighbours_topk_query(vector[qi], neigh_idx, neigh_d2, config);

        // Sum neighbor lrd
        float sum_lrd = 0.0f;
        for (int j = 0; j < k; j++) {
            sum_lrd += lrd[neigh_idx[j]];
        }
        float mean_lrd_neighbors = sum_lrd / (float)k;

        // Compute reachability density of the query using cached distances
        float sum_reach = 0.0f;
        for (int j = 0; j < k; j++) {
            int b = neigh_idx[j];
            float dist = sqrtf(neigh_d2[j]);  // only k sqrts per query
            float reach = fmaxf(kdist[b], dist);
            sum_reach += reach;
        }
        float lrd_query = 1.0f / (sum_reach / (float)k);

        scores[qi] = mean_lrd_neighbors / lrd_query;
    }

    // measure running time
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("Score Time taken: %f seconds\n", cpu_time_used);  // noisy, disable in production
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

#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int *neigh_idx = &all_neigh[i * k];
        float *neigh_d2 = &all_neigh_d2[i * k];
        tinyml_lof_k_neighbours_topk(i, neigh_idx, neigh_d2, config);

        // k-distance is sqrt of the k-th smallest squared distance
        config->k_distance[i] = sqrtf(neigh_d2[k - 1]);
    }

// 2) Compute lrd using cached neighbors and their d2 (no extra distance recomputation)
#pragma omp parallel for schedule(static)
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
