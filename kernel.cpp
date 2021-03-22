#include "kernel.h"

// Ackleys, Rastrigin, Sphere, Rosenbrock Sadle
float rosenbrock_function(float x[])
{
    float fitness = 0;
    for (int c = 0; c < DIMENSIONS - 1; c++)
    {
   	fitness += 100 * pow(x[c + 1] - (x[c] * x[c]), 2) + pow(1 - x[c], 2);
    }
    return fitness;
}

float ackleys_function(float x[])
{
   float first_sum = 0;
   float second_sum = 0;
   for (int c = 0; c < DIMENSIONS; c++)
   {	
	first_sum += pow(x[c], 2);
   	second_sum += cos(2.0 * phi * x[c]);
   }   
   float fitness = -20 * exp(-0.2 * sqrt(first_sum / 2)) - exp(second_sum / 2) + 20 + 2.7183;   
   return fitness;
}


float rastrigin_function(float x[])
{
   float fitness = 20;
   for (int c = 0; c < DIMENSIONS; c++)
   {
	fitness += pow(x[c], 2) - (10 * cos(2.0 * phi * x[c]));
   }
   return fitness;
} 


float sphere_function(float x[])
{
   float fitness = 0;
   for (int c = 0; c < DIMENSIONS; c++)
   {
	fitness += pow(x[c], 2);
   }
   return fitness;
}


float getRandom(float low, float high)
{
    return low + float(((high - low) + 1) * rand() / (RAND_MAX + 1.0));
}

// Generowanie losowych "wag" z przedziału od 0 do 1 wpływających na skalę oddziaływania
// personalnego i globalnego najlepszego wyniku cząsteczki
float getRandomLimited()
{
    return (float) rand() / (float) RAND_MAX;
}

void pso_cpu(float *particle_position, float *particle_velocity, float *personal_best_position, float *global_best_position)
{
    float particle_1[DIMENSIONS];
    float particle_2[DIMENSIONS];

    // Generowanie 
    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        for (int i = 0; i < PARTICLES * DIMENSIONS; i++)
        {
            float r1 = getRandomLimited();
            float r2 = getRandomLimited();
 
	    // Wzór PSO
            particle_velocity[i] = w * particle_velocity[i] + c1 * r1 
                    * (personal_best_position[i] - particle_position[i])
                    + c2 * r2 * (global_best_position[i % DIMENSIONS] - particle_position[i]);

            // Aktualizacja pozycji cząsteczki
            particle_position[i] = particle_position[i] + particle_velocity[i];
        }

        for (int i = 0; i < PARTICLES * DIMENSIONS; 
             i = i + DIMENSIONS)
        {

            for (int j = 0; j < DIMENSIONS; j++)
            {
		// Przygotowanie do porównania obecnej pozycji z najlepszą do tej pory dla tej cząsteczki
                particle_1[j] = particle_position[i + j];
                particle_2[j] = personal_best_position[i + j];
            }

	    // Porównanie obecnej pozycji z najlepszą do tej pory dla tej cząsteczki 
	    // (minimalizacja czyli najmniejsza jest najlepsza)
            if (rosenbrock_function(particle_1) < rosenbrock_function(particle_2))
            {
		// Jeśli tak to nadpisujemy jej najlepszą wersję obecną
                for (int k = 0; k < DIMENSIONS; k++)
                    personal_best_position[k] = particle_position[i + k];
		// I sprawdzamy czy może jest najlepsza w całej populacji cząsteczek
                if (rosenbrock_function(particle_2) < rosenbrock_function(global_best_position))
                {
		    // Jeśli tak to nadpisujemy również globalnie najlepszą cząsteczkę
                    for (int k = 0; k < DIMENSIONS; k++)
                        global_best_position[k] = personal_best_position[i + k];
                }
            }
        }
    }
}
