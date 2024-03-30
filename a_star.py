import heapq
from map import Map
from enum import Enum


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
        self.g = float('inf')  # custo do caminho do ponto inicial até este nó
        self.h = float('inf')  # estimativa do custo do caminho deste nó até o ponto final
        self.f = float('inf')  # custo total (f = g + h)
        self.parent = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, __value: object) -> bool:
        return self.x == __value.x and self.y == __value.y
    
class Grid:
    def __init__(self, map:Map, end_position:tuple):
        self.start_position = (end_position[0], end_position[1])
        self.end_position = (0,0)
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
        grid[self.end_position].type = Position.END
        
        return grid
    
    @property
    def start_node(self):
        return self.__g[self.start_position]
    
    @property
    def end_node(self):
        return self.__g[self.end_position]
    
    def get(self, x, y) -> Node:
        return self.__g[(x, y)]

class AStar:
    def __init__(self, map:Map, end_position:tuple):
        self.grid = Grid(map, end_position)
        self.open_set = []
        self.closed_set = set()
        self.start_node = self.grid.start_node
        self.end_node = self.grid.end_node
    
    def euclidean_distance(self, node1:Node, node2:Node):
        return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5
    
    def calculate_cost(self, current_node:Node, neighbor:Node):
        return current_node.g + neighbor.difficulty
    
    def get_neighbors(self, node):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x = node.x + dx
                new_y = node.y + dy
                if self.grid.min_x <= new_x < self.grid.max_x and self.grid.min_y <= new_y < self.grid.max_y and self.grid.get(new_x,new_y).type != Position.BLOCK:
                    neighbors.append(self.grid.get(new_x,new_y))
        return neighbors
    
    def reconstruct_path(self, current_node):
        path = []
        while current_node is not None:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent
        return path[::-1]
    
    def run(self):
        self.start_node.g = 0
        self.start_node.h = self.euclidean_distance(self.start_node, self.end_node)
        self.start_node.f = self.start_node.g + self.start_node.h
        heapq.heappush(self.open_set, self.start_node)
        
        while self.open_set:
            current_node = heapq.heappop(self.open_set)
            if current_node == self.end_node:
                return self.reconstruct_path(current_node)
            
            self.closed_set.add((current_node.x, current_node.y))
            for neighbor in self.get_neighbors(current_node):
                if (neighbor.x, neighbor.y) in self.closed_set:
                    continue
                
                tentative_g = current_node.g + self.calculate_cost(current_node, neighbor)
                if tentative_g < neighbor.g:
                    neighbor.parent = current_node
                    neighbor.g = tentative_g
                    neighbor.h = self.euclidean_distance(neighbor, self.end_node)
                    neighbor.f = neighbor.g + neighbor.h
                    heapq.heappush(self.open_set, neighbor)
        
        return None  # Não foi possível encontrar um caminho
