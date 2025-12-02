#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// The quadratic map g(x) = 4x(1-x)
double g(double x) {
    return 4.0 * x * (1.0 - x);
}

// Generate n observations from the quadratic map
void generate_time_series(double x0, int n, double *x) {
    x[0] = x0;
    for (int t = 0; t < n - 1; t++) {
        x[t + 1] = g(x[t]);
    }
}

int main() {
    double x0 = 0.3;
    int n = 100001;

    // Allocate memory
    double *x = (double *)malloc(n * sizeof(double));
    if (x == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Warm-up run
    generate_time_series(x0, 1000, x);

    // Timed run
    clock_t start = clock();
    generate_time_series(x0, n, x);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("C: Generated %d observations in %.6f seconds\n", n, elapsed);
    printf("C: %.0f iterations per second\n", n / elapsed);

    free(x);
    return 0;
}
