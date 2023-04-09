import random
import time
import csv
import matplotlib.pyplot as plt
#from knapsack import Knapsack

class Item:
    def __init__(self, valor, peso,frutas):
        self.valor = valor
        self.peso = peso
        self.frutas = frutas
class Mochila:
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
    

class AntColony:
    def __init__(self, mochila, num_ants, num_generations, alpha, beta, evaporation_rate):
        self.mochila = mochila
        self.num_ants = num_ants
        self.num_generations = num_generations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_trails = [1.0 for _ in range(len(mochila.item.valor))]

    def select_item(self, available_items):
        probabilities = []
        total = sum(self.pheromone_trails[i] ** self.alpha * (self.mochila.item.valor[i] / self.mochila.item.peso[i]) ** self.beta for i in available_items)

        for i in available_items:
            prob = (self.pheromone_trails[i] ** self.alpha * (self.mochila.item.valor[i] / self.mochila.item.peso[i]) ** self.beta) / total
            probabilities.append(prob)

        return random.choices(available_items, probabilities)[0]

    def construct_solution(self):
        solution = [0] * len(self.mochila.item.valor)
        available_items = list(range(len(self.mochila.item.valor)))
        weight = 0

        while available_items:
            item = self.select_item(available_items)
            if weight + self.mochila.item.peso[item] <= self.mochila.capacidad:
                solution[item] = 1
                weight += self.mochila.item.peso[item]
            available_items.remove(item)

        return solution

    def update_pheromones(self, best_solution):
        for i in range(len(self.pheromone_trails)):
            self.pheromone_trails[i] = (1 - self.evaporation_rate) * self.pheromone_trails[i]

        for i in range(len(best_solution)):
            if best_solution[i]:
                self.pheromone_trails[i] += self.mochila.value(best_solution)

    def run(self):
        best_solution = None
        best_value = 0
        convergence = []

        start_time = time.time()
        for generation in range(self.num_generations):
            solutions = [self.construct_solution() for _ in range(self.num_ants)]

            for solution in solutions:
                value = self.mochila.value(solution)
                if value > best_value:
                    best_solution = solution
                    best_value = value
                    best_iteration = self.num_ants * generation

            self.update_pheromones(best_solution)
            convergence.append(best_value)
        end_time = time.time()

        time_elapsed = end_time - start_time
        total_iterations = self.num_ants * self.num_generations
        return best_solution, convergence, time_elapsed, best_iteration, total_iterations

# Parámetros 
num_ants = 100
num_generations = 10
alpha = 0.5
beta = 0.5
evaporation_rate = 0.6

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

item=Item(valor, peso,frutas)
mochila = Mochila(item)

ant_colony = AntColony(mochila, num_ants, num_generations, alpha, beta, evaporation_rate)
mejorSolucion, convergence, diff_time, best_iteration, total_iterations = ant_colony.run()

print("Mejor solución encontrada:\n", mejorSolucion)
print("--------------------------------------------------")
print("Elementos en la mochila:\n", mochila.elementosEnLaMochila(mejorSolucion))
print("--------------------------------------------------")
print("Tiempo empleado para encontrar la solución:", diff_time, "segundos")
print("Valor total de la mochila:", mochila.value(mejorSolucion))
print("Peso total de la mochila:", mochila.pesoDeLaMochila(mejorSolucion))


# Visualiza el progreso del algoritmo
plt.plot(convergence)
plt.xlabel("Generación")
plt.ylabel("Mejor valor de la mochila ")
plt.title("Gráfica de Convergencia")
plt.show()