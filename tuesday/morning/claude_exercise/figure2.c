#include <stdio.h>
#include <stdlib.h>

#define N_STEPS 150

int main() {
    double trajectory[N_STEPS + 1];
    double x0, x_current;
    int t;
    FILE *fp;

    // Initial condition
    x0 = 0.3;
    trajectory[0] = x0;
    x_current = x0;

    // Generate time series using quadratic map g(x) = 4x(1-x)
    for (t = 1; t <= N_STEPS; t++) {
        x_current = 4.0 * x_current * (1.0 - x_current);
        trajectory[t] = x_current;
    }

    // Write to file
    fp = fopen("figure2_data_c.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    fprintf(fp, "# t x_t\n");
    for (t = 0; t <= N_STEPS; t++) {
        fprintf(fp, "%d %.10f\n", t, trajectory[t]);
    }
    fclose(fp);

    printf("Figure 2 data written to figure2_data_c.txt\n");

    return 0;
}
