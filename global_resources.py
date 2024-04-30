from matplotlib import pyplot as plt
import numpy as np
from map import Map
from vs.constants import VS

class GlobalResources:
    def __init__(self):
        self.map = Map()
        self.explorers = []
        self.rescuers = []
        self.victims = {}
        
    def add_explorer(self, explorer):
        self.explorers.append(explorer)
        return self
        
    def add_rescuer(self, rescuer):
        self.rescuers.append(rescuer)
        return self
    
    def __update_map(self):
        for explorer in self.explorers:
            map_data = explorer.map.map_data
            self.map.add_map_data(map_data)
        
    def all_explorers_finished(self):
        for explorer in self.explorers:
            if explorer.get_state() == VS.ACTIVE:
                return False
        return True
    
    def release_rescuers(self):
        for rescuer in self.rescuers:
            rescuer.start_work(self.map)
            
            print(f"ENV: {rescuer.NAME} is now active")
            
    def __add_victim(self):
        for explorer in self.explorers:
            self.victims.update(explorer.victims)
                
    def update_explorers_data(self):
        self.__update_map()
        self.__add_victim()

    def k_means_clustering(self, max_iterations=100):
        """ A method that groups the victims in 4 clusters
        @param victims: a dictionary with the victims found by the explorer
        @return a dictionary with the victims grouped by clusters"""

        # The victims are grouped in 4 clusters
        k = 4
        
        victims = []
        for chave, lista in self.victims.items():
                victims.append(lista[0])
        
        data = np.array(victims)

        rng = np.random.default_rng(0)
        centroids = data[rng.choice(range(len(data)), k, replace=False)]
        
        for _ in range(max_iterations):
            
            assignments = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
            new_centroids = np.array([data[assignments == i].mean(axis=0) for i in range(k)])
        
            if np.allclose(new_centroids, centroids):
                break
        
            centroids = new_centroids
            
        return assignments, centroids
    
    def plot_kmeans(self, assignments, centroids):
        cores = ['red', 'blue', 'green', 'purple']
        victims = []
        for chave, lista in self.victims.items():
                victims.append(lista[0])
        
        data = np.array(victims)
                    
        for i in range(4):
            cluster_points = data[assignments == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cores[i], label=f'Cluster {i+1}')
            plt.scatter(centroids[i, 0], centroids[i, 1], color=cores[i], marker='x', s=100)
                    
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo Y')
        
        min_x = min(key[0] for key in self.map.map_data.keys())
        max_x = max(key[0] for key in self.map.map_data.keys())
        min_y = min(key[1] for key in self.map.map_data.keys())
        max_y = max(key[1] for key in self.map.map_data.keys())
        plt.axis([min_x - 1, max_x + 1, max_y + 1, min_y - 1])
                    
        plt.scatter(0, 0, color='black', marker= '*' , label='Base')
                    
        plt.title('Gráfico dos Pontos do Cluster e Centroides')
                    
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def victims_by_cluster(self, assignments, cluster_id):
       
        victims = []
        for index, key in enumerate(self.victims):
           if assignments[index] == cluster_id:
               victims.append(self.victims[key])
        
        return victims
    