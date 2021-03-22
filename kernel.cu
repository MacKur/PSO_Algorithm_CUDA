#include <cuda_runtime.h>
#include <cuda.h>

#include "kernel.h"


__device__ float particle_1[DIMENSIONS];
__device__ float particle_2[DIMENSIONS];


__device__ float dev_rosenbrock_function(float x[])
{
   float fitness = 0;
   for (int c = 0; c < DIMENSIONS - 1; c++)
   {	
	fitness += 100 * pow(x[c + 1] - (x[c] * x[c]), 2) + pow(1 - x[c], 2);
   }
   return fitness;
}


__device__ float dev_ackleys_function(float x[])
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


__device__ float dev_rastrigin_function(float x[])
{
   float fitness = 20;
   for (int c = 0; c < DIMENSIONS; c++)
   {
	fitness += pow(x[c], 2) - (10 * cos(2.0 * phi * x[c]));
   }
   return fitness;
}


__device__ float dev_sphere_function(float x[])
{
   float fitness = 0;
   for (int c = 0; c < DIMENSIONS; c++)
   {
	fitness += pow(x[c], 2);
   }
   return fitness;
}


__global__ void kernelUpdateParticle(float *particle_position, float *particle_velocity, 
                                     float *personal_best_position, float *global_best_position, float r1, 
                                     float r2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= PARTICLES * DIMENSIONS)
        return;
    
    float rp = r1;
    float rg = r2;

    particle_velocity[i] = w * particle_velocity[i] + c1 * rp * (personal_best_position[i] - particle_position[i])
            + c2 * rg * (global_best_position[i % DIMENSIONS] - particle_position[i]);

    particle_position[i] = particle_position[i] +  particle_velocity[i];
}

__global__ void kernelUpdatePersonalBest(float *particle_position, float *personal_best_position, float *global_best_position)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= PARTICLES * DIMENSIONS || i % DIMENSIONS != 0)
        return;

    for (int j = 0; j < DIMENSIONS; j++)
    {
        particle_1[j] = particle_position[i + j];
        particle_2[j] = personal_best_position[i + j];
    }

    if (dev_rosenbrock_function(particle_1) < dev_rosenbrock_function(particle_2))
    {
        for (int k = 0; k < DIMENSIONS; k++)
            personal_best_position[i + k] = particle_position[i + k];
    }
}


extern "C" void pso_gpu(float *particle_position, float *particle_velocity, float *personal_best_position, 
                         float *global_best_position)
{
    int size = PARTICLES * DIMENSIONS;
    
    // Wskaźniki
    float *devPosition;
    float *devVelocity;
    float *devPersonalBest;
    float *devGlobalBest;
    
    float temp[DIMENSIONS];
        
    // Alokacja pamięci
    cudaMalloc((void**)&devPosition, sizeof(float) * size);
    cudaMalloc((void**)&devVelocity, sizeof(float) * size);
    cudaMalloc((void**)&devPersonalBest, sizeof(float) * size);
    cudaMalloc((void**)&devGlobalBest, sizeof(float) * DIMENSIONS);
    
    // Liczba wątków i bloków
    int threadsNum = 32;
    int blocksNum = PARTICLES / threadsNum; // (512 / 32) = 16
    
    // Pobranie cząsteczek z hosta na dev
    cudaMemcpy(devPosition, particle_position, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocity, particle_velocity, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPersonalBest, personal_best_position, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGlobalBest, global_best_position, sizeof(float) * DIMENSIONS, cudaMemcpyHostToDevice);
    
    // PSO
    for (int iter = 0; iter < ITERATIONS; iter++)
    {     
        // Aktualizacja pozycji i prędkości cząsteczek
        kernelUpdateParticle<<<blocksNum, threadsNum>>>(devPosition, devVelocity, 
                                                        devPersonalBest, devGlobalBest, 
                                                        getRandomLimited(), 
                                                        getRandomLimited());  
        // Aktualizacja najlepszej cząsteczki (lokalnej)
        kernelUpdatePersonalBest<<<blocksNum, threadsNum>>>(devPosition, devPersonalBest, devGlobalBest);
        
        // Aktualizacja najlepszej cząsteczki globalnie dla całej populacji
        cudaMemcpy(personal_best_position, devPersonalBest, 
                   sizeof(float) * PARTICLES * DIMENSIONS, 
                   cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < size; i += DIMENSIONS)
        {
            for(int k = 0; k < DIMENSIONS; k++)
                temp[k] = personal_best_position[i + k];
        
            if (rosenbrock_function(temp) < rosenbrock_function(global_best_position))
            {
                for (int k = 0; k < DIMENSIONS; k++)
                    global_best_position[k] = temp[k];
            }   
        }
        
        cudaMemcpy(devGlobalBest, global_best_position, sizeof(float) * DIMENSIONS, 
                   cudaMemcpyHostToDevice);
    }
    
    // Pobranie cząsteczek z dev na hosta
    cudaMemcpy(particle_position, devPosition, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_velocity, devVelocity, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(personal_best_position, devPersonalBest, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(global_best_position, devGlobalBest, sizeof(float) * DIMENSIONS, cudaMemcpyDeviceToHost); 
    
    
    cudaFree(devPosition);
    cudaFree(devVelocity);
    cudaFree(devPersonalBest);
    cudaFree(devGlobalBest);
}
