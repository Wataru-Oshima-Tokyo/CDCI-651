
from tkinter.tix import Tree
import pygame as pg
from heapq import *
from random import random
from collections import deque
import time

def get_circle(x, y):
    return (x * TILE + TILE // 2, y * TILE + TILE // 2), TILE // 4


def get_neighbours(x, y):
    check_neighbour = lambda x, y: True if 0 <= x < cols and 0 <= y < rows else False
    ways = [-1, 0], [0, -1], [1, 0], [0, 1], [-1, -1], [1, -1], [1, 1], [-1, 1] #8 ways to go
    return [(grid[y + dy][x + dx], (x + dx, y + dy)) for dx, dy in ways if check_neighbour(x + dx, y + dy)]


def get_click_mouse_pos():
    x, y = pg.mouse.get_pos()
    grid_x, grid_y = x // TILE, y // TILE
    pg.draw.circle(sc, pg.Color('red'), *get_circle(grid_x, grid_y))
    click = pg.mouse.get_pressed()
    return (grid_x, grid_y) if click[0] else False


def get_rect(x, y):
    return x * TILE + 1, y * TILE + 1, TILE - 2, TILE - 2


def heuristic(a, b):
   return abs(a[0] - b[0]) + abs(a[1] - b[1])




def dynamic_map(start, goal):
    create_time_start = time.time()
    global grid, graph
    grid=[[]]
    for row in range(rows):
        grid.append([])
        for col in range(cols):
            if random() < 0.2 and (col, row) != start and (col, row) != goal:
                grid[row].append(100)
            else:
                grid[row].append(0)
    create_time_end = time.time()
    return (create_time_end-create_time_start) #erase the time to create a new map


def bfs(start, goal, grpah, dynamic):
    queue = []
    heappush(queue, (0, start))
    cost_visited = {start: 0} 
    visited = {start: None}
    while queue:
        cur_cost, cur_node = heappop(queue)
        if cur_node == goal:
            break

        neighbours = graph[cur_node]
        for neighbour in neighbours:
            neigh_cost, neigh_node = neighbour
            new_cost = cost_visited[cur_node] + neigh_cost

            if neigh_node not in cost_visited or new_cost < cost_visited[neigh_node]:
                priority = new_cost
                heappush(queue, (priority, neigh_node))
                cost_visited[neigh_node] = new_cost
                visited[neigh_node] = cur_node
        if dynamic:
            start_+=dynamic_map(start, goal)
    return visited


def isRaise(point):
    pass

def expand(curr_cost, curr_point):
    isRaise =isRaise(curr_point)
    cost =0
    neighbours = graph[curr_point]
    for neighbour in neighbours:
        if isRaise:
            neigh_cost, neigh_node = neighbour
            for neigh_point in neigh_node:
                if neigh_point == curr_point:
                    new


def d_star(start, goal, graph, dynamic):
    queue =[]
    heappush(queue, (0,start))
    cost_visited = {start:0}
    visited = {start: None}
    while queue:
        cur_cost, cur_node = heappop(queue)
        expand(cur_cost, cur_node)


def dijkstra_astar(start, goal, graph, dynamic):
    global grid
    queue = []
    heappush(queue, (0, start))
    cost_visited = {start: 0}
    visited = {start: None}
    while queue:
        cur_cost, cur_node = heappop(queue)
        if cur_node == goal:
            break

        neighbours = graph[cur_node]
        for neighbour in neighbours:
            neigh_cost, neigh_node = neighbour
            new_cost = cost_visited[cur_node] + neigh_cost

            if neigh_node not in cost_visited or new_cost < cost_visited[neigh_node]:
                priority = new_cost + heuristic(neigh_node, goal)
                heappush(queue, (priority, neigh_node))
                cost_visited[neigh_node] = new_cost
                visited[neigh_node] = cur_node
        if dynamic:
            dynamic_map(start, goal)
    return visited


cols, rows = 100, 100
TILE = 5

pg.init()
sc = pg.display.set_mode([cols * TILE, rows * TILE])
clock = pg.time.Clock()
# set grid

# grid = [[100 if random() < 0.2 and (row !=0 and col !=7) else 0 for col in range(cols)] for row in range(rows)]
grid=[[]]
for row in range(rows):
    grid.append([])
    for col in range(cols):
        if random() < 0.2 and (row!=0 and col !=7):
            grid[row].append(100)
        else:
            grid[row].append(0)

# print(grid[0])


# adjacency dict
graph = {}
for y, row in enumerate(grid):
    for x, col in enumerate(row):
        graph[(x, y)] = graph.get((x, y), []) + get_neighbours(x, y)

start = (0, 7)
goal = start
queue = []
heappush(queue, (0, start))
visited = {start: None}
start_ = time.time()
 

while True:
    # fill screen
    sc.fill(pg.Color('black'))
    [[pg.draw.rect(sc, pg.Color('darkorange'), get_rect(x, y), border_radius=TILE // 5)
      for x, col in enumerate(row) if col] for y, row in enumerate(grid)]
    # bfs, get path to mouse click
    mouse_pos = get_click_mouse_pos() 
    if mouse_pos:
        start_ = time.time()
        visited = dijkstra_astar(start, mouse_pos, graph, dynamic=True)
        # visited = bfs(start, mouse_pos, graph,dynamic=True)
        goal = mouse_pos
        end_ = time.time()
        print("The time to find a path is ", (end_-start_))

    # draw path
    path_head, path_segment = goal, goal
    while path_segment and path_segment in visited:
        pg.draw.rect(sc, pg.Color('black'), get_rect(path_segment[0], path_segment[1]))
        pg.draw.circle(sc, pg.Color('blue'), *get_circle(*path_segment))
        path_segment = visited[path_segment]
    pg.draw.circle(sc, pg.Color('green'), *get_circle(*start))
    pg.draw.circle(sc, pg.Color('magenta'), *get_circle(*path_head))

    # pygame necessary lines
    [exit() for event in pg.event.get() if event.type == pg.QUIT]
    pg.display.flip()
    clock.tick(60)