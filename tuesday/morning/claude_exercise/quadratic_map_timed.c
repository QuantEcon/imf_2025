#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Quadratic map function: g(x) = 4x(1-x) */
double quadratic_map(double x) {
    return 4.0 * x * (1.0 - x);
}

/* Generate function data (computation only) */
void generate_function_data(double *x_vals, double *g_vals, int num_points) {
    for (int i = 0; i <= num_points; i++) {
        x_vals[i] = (double)i / num_points;
        g_vals[i] = quadratic_map(x_vals[i]);
    }
}

/* Generate trajectory data (computation only) */
void generate_trajectory(double x0, int n, double *trajectory) {
    double x = x0;
    for (int t = 0; t <= n; t++) {
        trajectory[t] = x;
        x = quadratic_map(x);
    }
}

/* Generate histogram data (computation only) */
void generate_histogram_data(double x0, int n, double *histogram_data) {
    double x = x0;
    for (int t = 0; t <= n; t++) {
        histogram_data[t] = x;
        x = quadratic_map(x);
    }
}

int main() {
    struct timespec start, end;
    double elapsed;

    // Allocate memory
    int num_points = 1000;
    double *x_vals = malloc((num_points + 1) * sizeof(double));
    double *g_vals = malloc((num_points + 1) * sizeof(double));

    int traj_n = 150;
    double *trajectory = malloc((traj_n + 1) * sizeof(double));

    int hist_n = 100000;
    double *histogram_data = malloc((hist_n + 1) * sizeof(double));

    // START TIMING
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Generate all data
    generate_function_data(x_vals, g_vals, num_points);
    generate_trajectory(0.3, traj_n, trajectory);
    generate_histogram_data(0.3, hist_n, histogram_data);

    // END TIMING
    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("C data generation time: %.6f seconds\n", elapsed);

    // Free memory
    free(x_vals);
    free(g_vals);
    free(trajectory);
    free(histogram_data);

    return 0;
}
