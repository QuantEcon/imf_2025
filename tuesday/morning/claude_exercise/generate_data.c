#include <stdio.h>
#include <stdlib.h>

// The quadratic map g(x) = 4x(1-x)
double g(double x) {
    return 4.0 * x * (1.0 - x);
}

// Generate data for Figure 1: The quadratic map
void generate_figure1_data(const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    // Generate 1000 points from 0 to 1
    int n = 1000;
    fprintf(fp, "x,g_x,line_45\n");

    for (int i = 0; i < n; i++) {
        double x = (double)i / (n - 1);
        double y = g(x);
        fprintf(fp, "%.10f,%.10f,%.10f\n", x, y, x);
    }

    fclose(fp);
    printf("Figure 1 data written to %s\n", filename);
}

// Generate data for Figure 2: Time series when x0 = 0.3
void generate_figure2_data(const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    double x0 = 0.3;
    int n = 150;

    fprintf(fp, "t,x_t\n");

    double x = x0;
    for (int t = 0; t < n; t++) {
        fprintf(fp, "%d,%.10f\n", t, x);
        x = g(x);
    }

    fclose(fp);
    printf("Figure 2 data written to %s\n", filename);
}

// Generate data for Figure 3: Histogram with 100,000 observations
void generate_figure3_data(const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    double x0 = 0.3;
    int n = 100001;  // 100,001 observations

    fprintf(fp, "x_t\n");

    double x = x0;
    for (int t = 0; t < n; t++) {
        fprintf(fp, "%.10f\n", x);
        x = g(x);
    }

    fclose(fp);
    printf("Figure 3 data written to %s\n", filename);
}

int main() {
    printf("Generating data for all three figures...\n\n");

    generate_figure1_data("data_figure1.csv");
    generate_figure2_data("data_figure2.csv");
    generate_figure3_data("data_figure3.csv");

    printf("\nAll data files generated successfully!\n");

    return 0;
}
