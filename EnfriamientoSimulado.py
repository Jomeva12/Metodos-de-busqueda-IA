import random
import math
import matplotlib.pyplot as plt
import time
import csv


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
    
class EnfriamientoSimulado:
    def __init__(self, mochila, initial_temperature, cooling_factor, max_iterations, max_temperature_levels):
        self.mochila = mochila
        self.initial_temperature = initial_temperature
        self.cooling_factor = cooling_factor
        self.max_iterations = max_iterations
        self.max_temperature_levels = max_temperature_levels


    def run(self):
        n = len(self.mochila.item.valor)
        current_solution = [0] * n
        best_solution = current_solution.copy()
        temperature = self.initial_temperature
        current_values = []
        best_solution_iterations = 0
        total_iterations = 0
    
        start_time = time.time()
        for level in range(self.max_temperature_levels):
            for iteration in range(self.max_iterations):
                neighbor = self.mochila.generate_neighbor(current_solution)
                delta =self.mochila.value(neighbor) - self.mochila.value(current_solution)
    
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_solution = neighbor
    
                    if self.mochila.value(current_solution) > self.mochila.value(best_solution):
                        best_solution = current_solution.copy()
                        best_solution_iterations = total_iterations
    
                total_iterations += 1
            current_values.append(self.mochila.value(current_solution))
            temperature *= self.cooling_factor
        end_time = time.time()
    
        time_elapsed = end_time - start_time
        
        return best_solution, current_values, time_elapsed, best_solution_iterations, total_iterations



initial_temperature = 5000
cooling_factor = 1.99
max_iterations = 1000
max_temperature_levels = 500
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


mEnfriamientoSimulado = EnfriamientoSimulado(mochila, initial_temperature, cooling_factor, max_iterations, max_temperature_levels)
mejorSolucion, current_values, diff_time, best_solution_iterations, total_iterations = mEnfriamientoSimulado.run()


print("Mejor solución encontrada:\n", mejorSolucion)
print("--------------------------------------------------")
print("Elementos en la mochila:\n", mochila.elementosEnLaMochila(mejorSolucion))
print("--------------------------------------------------")
print("Tiempo empleado para encontrar la solución:", diff_time, "segundos")
print("Valor total de la mochila:", mochila.value(mejorSolucion))
print("Peso total de la mochila:", mochila.pesoDeLaMochila(mejorSolucion))
print("Número de iteraciones para encontrar la mejor solución:", best_solution_iterations)
print("Número total de iteraciones:", total_iterations)


plt.plot(current_values)
plt.xlabel("Nivel de temperatura")
plt.ylabel("Valor de la mochila solución actual")
plt.title("Progreso del algoritmo de enfriamiento simulado")
plt.show()
