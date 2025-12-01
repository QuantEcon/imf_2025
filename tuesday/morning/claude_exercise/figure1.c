#include <stdio.h>
#include <stdlib.h>

#define N_POINTS 1000

int main() {
    double x_vals[N_POINTS];
    double g_vals[N_POINTS];
    double diagonal[N_POINTS];
    double dx;
    int i;
    FILE *fp;

    // Generate x values from 0 to 1
    dx = 1.0 / (N_POINTS - 1);
    for (i = 0; i < N_POINTS; i++) {
        x_vals[i] = i * dx;
        // Calculate g(x) = 4x(1-x)
        g_vals[i] = 4.0 * x_vals[i] * (1.0 - x_vals[i]);
        // 45 degree line
        diagonal[i] = x_vals[i];
    }

    // Write to file
    fp = fopen("figure1_data_c.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    fprintf(fp, "# x g(x) diagonal\n");
    for (i = 0; i < N_POINTS; i++) {
        fprintf(fp, "%.10f %.10f %.10f\n", x_vals[i], g_vals[i], diagonal[i]);
    }
    fclose(fp);

    printf("Figure 1 data written to figure1_data_c.txt\n");

    return 0;
}
