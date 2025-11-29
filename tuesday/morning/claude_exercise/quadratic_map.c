#include <stdio.h>
#include <stdlib.h>

/* Quadratic map function: g(x) = 4x(1-x) */
double quadratic_map(double x) {
    return 4.0 * x * (1.0 - x);
}

/* Generate trajectory data for time series */
void generate_trajectory(double x0, int n, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    double x = x0;
    fprintf(fp, "t,x\n");

    for (int t = 0; t <= n; t++) {
        fprintf(fp, "%d,%.15f\n", t, x);
        x = quadratic_map(x);
    }

    fclose(fp);
}

/* Generate data for plotting the function itself */
void generate_function_data(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    fprintf(fp, "x,g_x\n");

    int num_points = 1000;
    for (int i = 0; i <= num_points; i++) {
        double x = (double)i / num_points;
        double g_x = quadratic_map(x);
        fprintf(fp, "%.15f,%.15f\n", x, g_x);
    }

    fclose(fp);
}

/* Generate trajectory data for histogram (long time series) */
void generate_histogram_data(double x0, int n, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    double x = x0;
    fprintf(fp, "x\n");

    for (int t = 0; t <= n; t++) {
        fprintf(fp, "%.15f\n", x);
        x = quadratic_map(x);
    }

    fclose(fp);
}

int main() {
    printf("Generating quadratic map data...\n");

    /* Generate function data for Figure 1 */
    generate_function_data("quadratic_function.csv");
    printf("Generated quadratic_function.csv\n");

    /* Generate trajectory for Figure 2 (x0 = 0.3, n = 150 for visualization) */
    generate_trajectory(0.3, 150, "trajectory.csv");
    printf("Generated trajectory.csv\n");

    /* Generate long trajectory for Figure 3 (x0 = 0.3, n = 100000 for histogram) */
    generate_histogram_data(0.3, 100000, "histogram_data.csv");
    printf("Generated histogram_data.csv\n");

    printf("All data files generated successfully!\n");

    return 0;
}
