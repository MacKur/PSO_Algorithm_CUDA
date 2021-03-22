#include "kernel.h"

int main(int argc, char** argv)
{
    // Particle
    float particle_position[PARTICLES * DIMENSIONS];
    float particle_velocity[PARTICLES * DIMENSIONS];
    float personal_best_position[PARTICLES * DIMENSIONS];
    float global_best_position[DIMENSIONS];
    
    srand((unsigned) time(NULL));

    // Stworzenie cząsteczek do GPU
    for (int i = 0; i < PARTICLES * DIMENSIONS; i++)
    {
        particle_position[i] = getRandom(ROSENBROCK_LOWER_BOUND, ROSENBROCK_UPPER_BOUND);
        personal_best_position[i] = particle_position[i];
        particle_velocity[i] = 0;
    }

    for (int k = 0; k < DIMENSIONS; k++)
        global_best_position[k] = personal_best_position[k];        

    clock_t begin = clock();
    
    // ==================== GPU ======================= //
    pso_gpu(particle_position, particle_velocity, personal_best_position, global_best_position);
    
    clock_t end = clock();
    
    printf("==================== GPU =======================\n");
            
    printf("Processing time: %.6f [ms]\n", 
           (double)(end - begin) / CLOCKS_PER_SEC);
    
    
    // Nalepsza globalnie pozycja (problem minimalizacji)
    printf("Optimal solution:\n");
    for (int i = 0; i < DIMENSIONS; i++)
    printf("x%d = %f\n", i, global_best_position[i]);

    printf("Objective function value = %f\n", rosenbrock_function(global_best_position));
    
    //  ==================== GPU =======================  //
    

    // Stworzenie cząsteczek do CPU
    for (int i = 0; i < PARTICLES * DIMENSIONS; i++)
    {
        particle_position[i] = getRandom(ROSENBROCK_LOWER_BOUND, ROSENBROCK_UPPER_BOUND);
        personal_best_position[i] = particle_position[i];
        particle_velocity[i] = 0;
    }

    for (int k = 0; k < DIMENSIONS; k++)
        global_best_position[k] = personal_best_position[k];        

    begin = clock();
    
    //  ==================== CPU =======================  //
    pso_cpu(particle_position, particle_velocity, personal_best_position, global_best_position);
    
    end = clock();
    
    printf("==================== CPU =======================\n");
            
    printf("Processing time: %.6f [ms]\n", 
           (double)(end - begin) / CLOCKS_PER_SEC);
    
    
    // Najlepsza globalnie pozycja (problem minimalizacji) 
    printf("Optimal solution:\n");
    for (int i = 0; i < DIMENSIONS; i++)
        printf("x%d = %f\n", i, global_best_position[i]);

    printf("Objective function value = %f\n", rosenbrock_function(global_best_position));
    
    //  ==================== CPU =======================  //

    return 0;
}
