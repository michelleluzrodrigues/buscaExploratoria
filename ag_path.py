from enum import Enum
import random
import matplotlib.pyplot as plt

from map import Map

class Position(Enum):
    EMPTY = ' '
    START = 'S'
    END = 'E'
    BLOCK = 'B'
    
class Node:
    def __init__(self, position:tuple, difficulty = 1, type = Position.EMPTY):
        self.x = position[0]
        self.y = position[1]
        self.difficulty = difficulty
        self.type = type
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, __value: object) -> bool:
        return self.x == __value.x and self.y == __value.y
    
class Grid:
    def __init__(self, map:Map, start_position:tuple):
        self.start_position = (start_position[0], start_position[1])
        self.__g = self.generate_grid(map)
        
    def generate_grid(self, map:Map):
        
        data = map.map_data
        
        self.min_x = min(key[0] for key in data.keys())
        self.max_x = max(key[0] for key in data.keys())
        self.min_y = min(key[1] for key in data.keys())
        self.max_y = max(key[1] for key in data.keys())
        
        self.size_x = self.max_x + abs(self.min_x) + 1
        self.size_y = self.max_y + abs(self.min_y) + 1
        
        grid = {}
        for y in range(self.min_y, self.max_y + 1):
            for x in range(self.min_x, self.max_x + 1):
                if (x, y) in data:
                    grid[(x,y)] = Node((x, y), data[(x, y)][0])
                else:
                    grid[(x,y)] = Node((x, y), 100, Position.BLOCK)
        
        grid[self.start_position].type = Position.START
        
        return grid
    
    @property
    def start_node(self):
        return self.__g[self.start_position]
    
    @property
    def end_node(self):
        return self.__g[self.end_position]
    
    def get(self, x, y) -> Node:
        return self.__g[(x, y)]
    
class AGPath:
    NUM_EXECUCOES = 10        # número de execuções
    TAM_POP = 32              # tamanho da população
    MAX_GERACOES = 30        # máximo de gerações por execução
    PROB_CROSSOVER = 0.75     # probabilidade de cruzamento entre dois indivíduos
    PROB_MUTACAO = 0.04       # probabilidade de mutação sobre um indivíduo
    
    def __init__(self, mapa: Map, energia_inicial, posicao_inicial:tuple, lista_vitimas):
        self.energia_inicial = energia_inicial
        self.posicao_inicial = posicao_inicial
        self.lista_vitimas = self.trata_list_vitimas(lista_vitimas)
        self.mapa = Grid(mapa, posicao_inicial)
        self.populacao = [self.criar_caminho_aleatorio() for _ in range(self.TAM_POP)]
        self.melhor_caminho = None
        self.melhor_fitness = float('-inf')
        
    def criar_caminho_aleatorio(self):
        # Mapa como lista de listas, onde cada elemento pode ser um custo ou um valor alto para bloqueio
        pos = self.posicao_inicial
        path = []
        energy = self.energia_inicial
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # 8 possíveis movimentos

        while energy > 0:
            # Escolhe um movimento aleatório entre os vizinhos válidos
            valid_moves = [(d[0], d[1]) for d in directions
                        if self.mapa.min_x <= (pos[0] + d[0]) < self.mapa.max_x and self.mapa.min_y <= (pos[1] + d[1]) < self.mapa.max_y and self.mapa.get(pos[0] + d[0],pos[1] + d[1]).type != Position.BLOCK]
            
            if not valid_moves:
                break  # Sem movimentos possíveis, termina o caminho

            # Escolhe um movimento aleatório dentre os válidos
            direct = random.choice(valid_moves)
            pos = (pos[0] + direct[0], pos[1] + direct[1])
            path.append(direct)
            energy -= self.mapa.get(pos[0],pos[1]).difficulty  # Subtrai o custo do movimento da energia

        return path

    def executar_ag(self):
        for _ in range(self.MAX_GERACOES):
            # Avaliação de fitness para cada caminho na população
            fitness_scores = [self.calcular_fitness(caminho) for caminho in self.populacao]
            novos_pais = self.selecionar(fitness_scores)
            novos_filhos = self.aplicar_crossover(novos_pais)
            self.aplicar_mutacao(novos_filhos)
            self.populacao = novos_filhos  # Nova população para próxima geração

            # Atualizar o melhor caminho encontrado
            max_fitness = max(fitness_scores)
            if max_fitness > self.melhor_fitness:
                self.melhor_fitness = max_fitness
                self.melhor_caminho = self.populacao[fitness_scores.index(max_fitness)]



    def calcular_fitness(self, caminho):
        # Inicialização das variáveis de fitness
        energia_usada = 0
        vitimas_resgatadas = 0
        gravidade_resgatadas = 0
        ultimo_idx_resgate = None
        
        pos = self.posicao_inicial
        
        pos_vitima = []

        # Verificar cada posição no caminho
        for idx, direct in enumerate(caminho):
            pos = (pos[0] + direct[0], pos[1] + direct[1])
            if self.mapa.min_x > pos[0]:
                caminho[idx] = (caminho[idx][0] + 1, caminho[idx][1])
                pos = (pos[0] + 1, pos[1])
            elif pos[0] > self.mapa.max_x:
                caminho[idx] = (caminho[idx][0] - 1, caminho[idx][1])
                pos = (pos[0] - 1, pos[1])
            elif self.mapa.min_y > pos[1]:
                caminho[idx] = (caminho[idx][0], caminho[idx][1] + 1)
                pos = (pos[0], pos[1] + 1)
            elif pos[1] > self.mapa.max_y:
                caminho[idx] = (caminho[idx][0], caminho[idx][1] - 1) 
                pos = (pos[0], pos[1] - 1)
            #energia_usada += self.mapa.get(pos[0],pos[1]).difficulty
            for v_pos, gravidade in self.lista_vitimas:
                if v_pos == pos and v_pos not in pos_vitima:
                    vitimas_resgatadas += 1
                    gravidade_resgatadas += gravidade
                    ultimo_idx_resgate = idx  # Atualizar o índice do último resgate
                    pos_vitima.append(v_pos)

        # Se todas as vítimas foram resgatadas, ajustar o cálculo da energia até a última vítima
        if vitimas_resgatadas == len(self.lista_vitimas) and ultimo_idx_resgate is not None:
            pos = self.posicao_inicial
            for idx, direct in enumerate(caminho[:ultimo_idx_resgate + 1]):
                pos = (pos[0] + direct[0], pos[1] + direct[1])
                if self.mapa.min_x > pos[0]:
                    caminho[idx] = (caminho[idx][0] + 1, caminho[idx][1])
                    pos = (pos[0] + 1, pos[1])
                elif pos[0] > self.mapa.max_x:
                    caminho[idx] = (caminho[idx][0] - 1, caminho[idx][1])
                    pos = (pos[0] - 1, pos[1])
                elif self.mapa.min_y > pos[1]:
                    caminho[idx] = (caminho[idx][0], caminho[idx][1] + 1)
                    pos = (pos[0], pos[1] + 1)
                elif pos[1] > self.mapa.max_y:
                    caminho[idx] = (caminho[idx][0], caminho[idx][1] - 1) 
                    pos = (pos[0], pos[1] - 1)
                energia_usada += self.mapa.get(pos[0],pos[1]).difficulty

        energia_restante = self.energia_inicial - energia_usada

        # Fitness é uma combinação de vítimas resgatadas, sua gravidade e a energia restante
        fitness = (vitimas_resgatadas * 100) + (gravidade_resgatadas * 100) + energia_restante
        return fitness

    def aplicar_mutacao(self, filhos):
        for filho in filhos:
            if random.random() < self.PROB_MUTACAO:
                pos_mut = random.randint(0, len(filho) - 1)
                movimentos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                new_x, new_y = random.choice(movimentos)
                filho[pos_mut] = (new_x, new_y)

    def selecionar(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        selecao_probs = [f / total_fitness for f in fitness_scores]
        selecionados = random.choices(range(len(fitness_scores)), weights=selecao_probs, k=len(fitness_scores))
        return selecionados

    def imprimir_resultados(self):
        print(f"Melhor fitness: {self.melhor_fitness}")
        print(f"Melhor caminho: {self.melhor_caminho}")
    
    def aplicar_crossover(self, pais):
        filhos = []
        for i in range(0, len(pais), 2):
            pai1, pai2 = self.populacao[pais[i]], self.populacao[pais[i+1]]
            if random.random() < self.PROB_CROSSOVER:
                ponto_corte = random.randint(1, len(pai1) - 1)
                filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
                filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]
                filhos.extend([filho1, filho2])
            else:
                filhos.extend([pai1, pai2])
        return filhos
    
    def generate_grid(self, map:Map):
        
        data = map.map_data
        
        self.min_x = min(key[0] for key in data.keys())
        self.max_x = max(key[0] for key in data.keys())
        self.min_y = min(key[1] for key in data.keys())
        self.max_y = max(key[1] for key in data.keys())
        
        self.size_x = self.max_x + abs(self.min_x) + 1
        self.size_y = self.max_y + abs(self.min_y) + 1
        
        grid = {}
        for y in range(self.min_y, self.max_y + 1):
            for x in range(self.min_x, self.max_x + 1):
                if (x, y) in data:
                    grid[(x,y)] = Node((x, y), data[(x, y)][0])
                else:
                    grid[(x,y)] = Node((x, y), 100, Position.BLOCK)
        
        grid[self.posicao_inicial].type = Position.START
        
        return grid
    
    def trata_list_vitimas(self, list_vitimas):
        vitimas = []
        for vitima in list_vitimas:
            vitimas.append((vitima[0], vitima[1][6]))
        
        return vitimas