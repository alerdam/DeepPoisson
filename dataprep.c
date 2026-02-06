#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Constants for grid size and simulation depth */
#define SIZE 50
#define MAX_ITER 2000 
#define NUM_SAMPLES 1000 

/**
 * Solves the Poisson equation: Del^2 V = f 
 * Using Finite Difference Method (Jacobi Iteration)
 */
void solve_poisson_c(float V[SIZE][SIZE], float f[SIZE][SIZE]) {
    float V_new[SIZE][SIZE];
    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 1; i < SIZE - 1; i++) {
            for (int j = 1; j < SIZE - 1; j++) {
                /* Finite Difference approximation of the Laplacian */
                V_new[i][j] = 0.25f * (V[i+1][j] + V[i-1][j] + V[i][j+1] + V[i][j-1] - f[i][j]);
            }
        }
        /* Update the main grid with computed values */
        for (int i = 1; i < SIZE - 1; i++) {
            for (int j = 1; j < SIZE - 1; j++) V[i][j] = V_new[i][j];
        }
    }
}

int main() {
    srand(time(NULL));
    FILE *fx = fopen("x_train.bin", "wb"); /* Input: Boundaries + Source */
    FILE *fy = fopen("y_train.bin", "wb"); /* Target: Full Field Solution */

    printf("Generating %d physics scenarios...\n", NUM_SAMPLES);

    for (int s = 0; s < NUM_SAMPLES; s++) {
        float V[SIZE][SIZE] = {0};
        float f[SIZE][SIZE] = {0};
        float input_frame[SIZE][SIZE] = {0};

        /* 1. Generate Random Boundary Conditions */
        float v_top = (float)(rand() % 100);
        float v_bot = (float)(rand() % 100);
        for(int j=0; j<SIZE; j++) {
            V[0][j] = v_top; 
            V[SIZE-1][j] = v_bot;
        }

        /* 2. Generate Random Point Source */
        int src_x = 1 + rand() % (SIZE - 2);
        int src_y = 1 + rand() % (SIZE - 2);
        f[src_x][src_y] = (float)(rand() % 50) - 25.0f; 

        /* Save initial state as Input (X) */
        for(int i=0; i<SIZE; i++) {
            for(int j=0; j<SIZE; j++) input_frame[i][j] = V[i][j] + f[i][j];
        }

        /* 3. Compute Ground Truth */
        solve_poisson_c(V, f);

        /* 4. Stream to binary files */
        fwrite(input_frame, sizeof(float), SIZE * SIZE, fx);
        fwrite(V, sizeof(float), SIZE * SIZE, fy);
    }

    fclose(fx); fclose(fy);
    printf("Success: x_train.bin and y_train.bin generated.\n");
    return 0;
}