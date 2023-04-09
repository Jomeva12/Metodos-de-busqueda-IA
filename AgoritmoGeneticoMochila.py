import random
import matplotlib.pyplot as plt
import time

import random
import csv
class Item:
    def __init__(self, valor, peso,frutas):
        self.valor = valor
        self.peso = peso
        self.frutas = frutas
class Mochila:
    capacidaa=0
    def __init__(self,item):       
        self.capacidad=30000
        self.item=item
     
       
    def value(self, solution):
        valor_total = 0
        total_weight = 0
        for i in range(len(solution)):
            if solution[i]:
                valor_total += self.item.valor[i]
                total_weight += self.item.peso[i]
         
        return valor_total if total_weight <= self.capacidad else 0
    
    def pesoDeLaMochila(self, solution):
        valor_total = 0
        total_weight = 0
        for i in range(len(solution)):
            if solution[i]==1 and total_weight <= self.capacidad:
                valor_total += self.item.valor[i]
                total_weight += self.item.peso[i]
         
        return total_weight 
    def elementosEnLaMochila(self, solution):        
        elemento = []
        for i in range(len(solution)):
            if solution[i]==1:               
                elemento.append( self.item.frutas[i])         
        return elemento 

    def generate_neighbor(self, solution):
        index = random.randint(0, len(solution) - 1)
        neighbor = solution.copy()
        neighbor[index] = 1 - neighbor[index]
        return neighbor
    

class AlgoritmoGenetico:
    def __init__(self, mochila, population_size, mutation_rate, crossover_rate, max_generations):
        self.mochila = mochila
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations

    def create_individual(self):
       # n = len(self.mochila.valor)
        n = len(self.mochila.item.valor)
        individual = [random.randint(0, 1) for _ in range(n)]
        return individual

    def create_population(self):
        population = [self.create_individual() for _ in range(self.population_size)]
        return population

    def fitness(self, individual):
        return self.mochila.value(individual)

    def mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                mutated_individual[i] = 1 - mutated_individual[i]
        return mutated_individual

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def selection(self, population):
        return random.choices(population, weights=[self.fitness(individual) for individual in population], k=2)

    def proceso(self):
        population = self.create_population()
        best_individual = max(population, key=self.fitness)
        convergence = []
        best_solution_generation = 0
        
        tiempoInicio = time.time()
        for generation in range(self.max_generations):
            new_population = []

            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection(population)
                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            population = new_population
            mejorIndividioActual = max(population, key=self.fitness)
            if self.fitness(mejorIndividioActual) > self.fitness(best_individual):
                best_individual = mejorIndividioActual
                best_solution_generation = generation

            convergence.append(self.fitness(mejorIndividioActual))
            
        c = time.time()
        tiempoFin = time.time()
        diff_time = tiempoFin - tiempoInicio
        return best_individual, convergence, diff_time, best_solution_generation
valor=[]
peso=[]
frutas=[]
with open('datos.csv', newline='') as archivo_csv:
    lector_csv = csv.reader(archivo_csv, delimiter=';')
    siguiente = False
    for fila in lector_csv:
        if not siguiente:
            siguiente = True
            continue
        
        valor.append(int(fila[1]))
        peso.append(int(fila[2]))
        frutas.append((fila[3]))
        

# Parámetros
poblacion = 100
tasaMutacion = 0.1
tasaCruce = 0.8
generacionesMax = 1000

item=Item(valor, peso,frutas)
mochila = Mochila(item)
ga = AlgoritmoGenetico(mochila, poblacion, tasaMutacion, tasaCruce, generacionesMax)
mejorSolucion, convergence, diff_time, best_solution_generation = ga.proceso()

print("Mejor solución encontrada:\n", mejorSolucion)
print("--------------------------------------------------")
print("Elementos en la mochila:\n", mochila.elementosEnLaMochila(mejorSolucion))
print("--------------------------------------------------")
print("Tiempo empleado para encontrar la solución:", diff_time, "segundos")
print("Valor total de la mochila:", mochila.value(mejorSolucion))
print("Peso total de la mochila:", mochila.pesoDeLaMochila(mejorSolucion))
print("Número de iteraciones para encontrar la mejor solución:", best_solution_generation)
print("Número total de iteraciones:", generacionesMax * poblacion)


# Visualiza el progreso del algoritmo
plt.plot(convergence)
plt.xlabel("Generación")
plt.ylabel("Mejor valor de la mochila ")
plt.title("Gráfica de Convergencia")
plt.show()