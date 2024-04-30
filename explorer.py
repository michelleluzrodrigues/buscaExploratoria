# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from global_resources import GlobalResources
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
import time
from a_star import AStar

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc, id):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.time_to_comeback = math.ceil(self.TLIM * 0.6)  # set the time to come back to the base
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
                                   
        self.visited = set()       # a set to store the visited cells
        self.id = id
        self.NAME = f"Explorer_{id}"
        
        self.dfs_move_order = self.translocate_list([(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)], id)
        
        self.dfs_return = []
        
        self.return_way = []
        self.force_return = False

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.visited.add((self.x, self.y)) # add the base to the visited cells
        
    def add_global_resources(self, global_resources: GlobalResources):
        self.global_resources = global_resources
        
    def translocate_list(self, list, shift):
        return list[-(shift * 2):] + list[:-(shift * 2)]

    def DFS(self):
        """ Depth-first search algorithm to explore the environment """
        # Check the neighborhood walls and grid limits
        obstacles = self.translocate_list(self.check_walls_and_lim(), self.id)
        
        # Loop until a CLEAR position is found
        for i in range(8):
            if obstacles[i] == VS.CLEAR:
                dx, dy = self.dfs_move_order[i]
                if (self.x + dx, self.y + dy) not in self.visited:
                    self.visited.add((self.x + dx, self.y + dy))
                    self.dfs_return = []
                    return dx, dy
                    
        # if there is no CLEAR position, the agent should go back
        if (self.dfs_return == []):
            self.dfs_return = self.walk_stack.items.copy()
        lx, ly = self.dfs_return.pop()
        
        return lx * -1, ly * -1
        
    def explore(self):
        # get the next position using DFS
        dx, dy = self.DFS()
        
        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy          

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        # dx, dy = self.walk_stack.pop()coming back
        # dx = dx * -1
        # dy = dy * -1
        dx,dy = self.return_way.pop(0)
        dx = dx - self.x
        dy = dy - self.y

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
            
    def back_step(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            print(f"{self.NAME}: couldn't calculate the path at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
       
    def calculate_way_cust(self, way):
        total_difficulty = 0
        for node in way:
            total_difficulty += self.map.get(node)[0]
            
        return total_difficulty
         
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        #time.sleep(0.1)

        if self.get_rtime() <= self.time_to_comeback and len(self.return_way) == 0:
            
            astar = AStar(self.map, (self.x, self.y))
            come_back_way = astar.run()
            
            if come_back_way == None:
                self.back_step()
                self.force_return = True
            else:
                cust = self.calculate_way_cust(come_back_way)
                self.force_return = False
            
                if self.get_rtime() <= cust * 1.6:
                    come_back_way.pop(0) # remove the base from the way because the agent is already there
                    self.return_way = come_back_way
                else:
                    self.time_to_comeback = cust + self.get_rtime() / 2
        
        if len(self.return_way) == 0 and not self.force_return:
            self.explore()
            return True
        elif not self.force_return:
            self.come_back()
        
        if self.x == 0 and self.y == 0:
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME}: rtime {self.get_rtime()}")
            #input(f"{self.NAME}: type [ENTER] to proceed")
            self.set_state(VS.IDLE)
            
            if (self.global_resources.all_explorers_finished()):
                self.global_resources.update_explorers_data()
                assignments, centroids = self.global_resources.k_means_clustering()
                #self.global_resources.plot_kmeans(assignments, centroids)
                
                for i in range(4):
                    vit = self.global_resources.victims_by_cluster(assignments, i)
                    self.global_resources.rescuers[i].add_victim(vit)
                
                self.global_resources.release_rescuers()
            return False
        
        return True

