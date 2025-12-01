#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_STEPS 100000

int main() {
    double trajectory[N_STEPS + 1];
    double x0, x_current;
    int t;
    FILE *fp;
    clock_t start, end;
    double elapsed_time;

    // Initial condition
    x0 = 0.3;
    trajectory[0] = x0;
    x_current = x0;

    // Start timer
    start = clock();

    // Generate long time series using quadratic map g(x) = 4x(1-x)
    for (t = 1; t <= N_STEPS; t++) {
        x_current = 4.0 * x_current * (1.0 - x_current);
        trajectory[t] = x_current;
    }

    // End timer
    end = clock();
    elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Write to file
    fp = fopen("figure3_data_c.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    fprintf(fp, "# x_t\n");
    for (t = 0; t <= N_STEPS; t++) {
        fprintf(fp, "%.10f\n", trajectory[t]);
    }
    fclose(fp);

    printf("Figure 3 data written to figure3_data_c.txt\n");
    printf("Total observations: %d\n", N_STEPS + 1);
    printf("Execution time: %.6f seconds\n", elapsed_time);

    return 0;
}
