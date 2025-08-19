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

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#define LOF_VECTOR(i, config) &(config->data[i * config->vector_size])
#define TINYML_MAX_DISTANCE 99999.0

void tinyml_lof_init(tinyml_lof_config_t *config) {
}

float tinyml_lof_normal_distance_vec(float *vec_a, float *vec_b, int len) {
    float dist = 0;

    for (int i = 0; i < len; i++) {
        dist += pow(vec_a[i] - vec_b[i], 2);
        // dist += abs(vec_a[i] - vec_b[i]);
    }
    // return dist;
    return sqrt(dist);
}

// Helper struct to store distance and index
typedef struct {
    float dist;
    int index;
} distance_t;

// Comparison function for qsort
int compare_distances(const void *a, const void *b) {
    float dist_a = ((distance_t *)a)->dist;
    float dist_b = ((distance_t *)b)->dist;
    if (dist_a < dist_b)
        return -1;
    if (dist_a > dist_b)
        return 1;
    return 0;
}

// void tinyml_lof_k_neighbours_vec(float *vector, int *neighbours, tinyml_lof_config_t *config) {
//     // This assumes config->n is not excessively large to cause stack overflow.
//     // For very large n, allocate `all_distances` on the heap instead.
//     distance_t all_distances[config->n];

//     // 1. Calculate distance to all points
//     for (int i = 0; i < config->n; i++) {
//         all_distances[i].dist = tinyml_lof_normal_distance_vec(vector, LOF_VECTOR(i, config), config->vector_size);
//         all_distances[i].index = i;
//     }

//     // 2. Sort distances in ascending order
//     qsort(all_distances, config->n, sizeof(distance_t), compare_distances);

//     // 3. Get the first k neighbours
//     for (int k = 0; k < config->parameter_k; k++) {
//         neighbours[k] = all_distances[k].index;
//     }
// }

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
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        scores[i] = tinyml_lof_score(vector[i], config);
    }
}

void tinyml_lof_learn(tinyml_lof_config_t *config) {
    int neighbours[10];

    if (config->n < (config->parameter_k + 1))
        return;

    for (int i = 0; i < config->n; i++) {
        tinyml_lof_k_neighbours(i, neighbours, config);

        // k-distance calculation
        config->k_distance[i] = tinyml_lof_normal_distance_vec(LOF_VECTOR(i, config), LOF_VECTOR(neighbours[config->parameter_k - 1], config), config->vector_size);
        // lrd distance calculation
        config->lrd[i] = tinyml_lof_reachability_density(LOF_VECTOR(i, config), neighbours, config);
    }
}
