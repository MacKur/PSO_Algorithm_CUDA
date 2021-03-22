#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const int ITERATIONS = 50000; 			// Liczba iteracji algorytmu
const int PARTICLES = 512;			// Liczba cząsteczek
const int DIMENSIONS = 2;			// Liczba wymiarów

// Górne i dolne granice dla różnych funkcji 
const float ACKLEYS_LOWER_BOUND = -32.0f;
const float ACKLEYS_UPPER_BOUND = 32.0f;
const float RASTRIGIN_LOWER_BOUND = -5.12f;	
const float RASTRIGIN_UPPER_BOUND = 5.12f;	
const float SPHERE_LOWER_BOUND = -5.12f;
const float SPHERE_UPPER_BOUND = 5.12f;
const float ROSENBROCK_LOWER_BOUND = -2.048f;
const float ROSENBROCK_UPPER_BOUND = 2.048f;	
 
const float w = 0.75; 				// Omega
const float c1 = 1; 				// Cognitive constant
const float c2 = 2; 				// Social constant
const float phi = 3.1415;

float getRandom(float low, float high);
float getRandomLimited();
float rosenbrock_function(float x[]);
float ackleys_function(float x[]);
float rastrigin_function(float x[]);
float sphere_function(float x[]); 

extern "C" void pso_gpu(float *particle_position, float *particle_velocity, float *personal_best_position, float *global_best_position);
void pso_cpu(float *particle_position, float *particle_velocity, float *personal_best_position, float *global_best_position);
