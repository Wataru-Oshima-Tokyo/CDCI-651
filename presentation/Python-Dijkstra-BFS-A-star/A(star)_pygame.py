import pygame as pg
from heapq import *
import numpy as np
from PIL import Image

def get_circle(x, y):
    return (x * TILE + TILE // 2, y * TILE + TILE // 2), TILE // 4


def get_rect(x, y):
    return x * TILE + 1, y * TILE + 1, TILE - 2, TILE - 2


def get_next_nodes(x, y, grid):
    check_next_node = lambda x, y: True if 0 <= x < cols and 0 <= y < rows else False
    ways = [-1, 0], [0, -1], [1, 0], [0, 1]
    return [(grid[y + dy][x + dx], (x + dx, y + dy)) for dx, dy in ways if check_next_node(x + dx, y + dy)]


def heuristic(a, b):
   return abs(a[0] - b[0]) + abs(a[1] - b[1])


file_path = "/Users/wataruoshima/CSCI651/presentation/Python-Dijkstra-BFS-A-star/files/map_sh.pgm"
file = "/Users/wataruoshima/CSCI651/presentation/Python-Dijkstra-BFS-A-star/img/1.png"
image = Image.open(file_path)
print(image.format)
print(image.size)
print(image.mode)
np_img = np.array(image)


# cols, rows = 23, 13
# rows ,cols = image.size[0], image.size[1]
# for i in range(rows):
#     print(np_img[0][i])

# grid

# grid = ['22222222222222222222212',
#         '22222292222911112244412',
#         '22444422211112911444412',
#         '24444444212777771444912',
#         '24444444219777771244112',
#         '92444444212777791192144',
#         '22229444212777779111144',
#         '11111112212777772771122',
#         '27722211112777772771244',
#         '27722777712222772221244',
#         '22292777711144429221244',
#         '22922777222144422211944',
#         '22222777229111111119222']



grid = ['2222222222222222222221222292777711144429221244',
        '2222229222291111224441222222777229111111119222',
        '2244442221111291144441227722777712222772221244',
        '2444444421277777144491211111112212777772771122',
        '2444444421977777124411227722777712222772221244',
        '9244444421277779119214411111112212777772771122',
        '2222944421277777911114422222777229111111119222',
        '1111111221277777277112222229444212777779111144',
        '2772221111277777277124424444444212777771444912',
        '2772277771222277222124492444444212777791192144',
        '2229277771114442922124424444444212777771444912',
        '2292277722214442221194422222222222222222222212',
        '2222277722911111111922222444422211112911444412']

rows ,cols = len(grid[0]), len(grid)

TILE = 0.5

pg.init()
sc = pg.display.set_mode([cols * TILE, rows * TILE])
clock = pg.time.Clock()

#change the format here
# grid =[]
# for i in range(cols):
#     temp =[]
#     for j in range(rows):
#         if np_img[i][j] >205:
#             temp.append(1)
#         else:
#             temp.append(0)
#     grid.append(temp)
# grid = [[int(char) for char in string ] for string in grid]
# print(grid)
# dict of adjacency lists
graph = {}
for y, row in enumerate(grid):
    for x, col in enumerate(row):
        graph[(x, y)] = graph.get((x, y), []) + get_next_nodes(x, y,grid)

graph = {}
for y, row in enumerate(np_img):
    for x, col in enumerate(row):
        graph[(x, y)] = graph.get((x, y), []) + get_next_nodes(x, y, grid)
        # print("x", x)
        # print("col", col)
    if y %100 ==0:
        print(y)

# BFS settings
start = (0, 7)
goal = (rows, cols)
queue = []
heappush(queue, (0, start))
cost_visited = {start: 0}
visited = {start: None}

bg = pg.image.load(file_path).convert()
bg = pg.transform.scale(bg, (cols * TILE, rows * TILE))

while True:
    # fill screen
    sc.blit(bg, (0, 0))
    # draw BFS work
    [pg.draw.rect(sc, pg.Color('forestgreen'), get_rect(x, y), 1) for x, y in visited]
    [pg.draw.rect(sc, pg.Color('darkslategray'), get_rect(*xy)) for _, xy in queue]
    pg.draw.circle(sc, pg.Color('purple'), *get_circle(*goal))

    # Dijkstra logic
    if queue:
        cur_cost, cur_node = heappop(queue)
        if cur_node == goal:
            queue = []
            continue

        next_nodes = graph[cur_node]
        for next_node in next_nodes:
            neigh_cost, neigh_node = next_node
            new_cost = cost_visited[cur_node] + neigh_cost

            if neigh_node not in cost_visited or new_cost < cost_visited[neigh_node]:
                priority = new_cost + heuristic(neigh_node, goal)
                heappush(queue, (priority, neigh_node))
                cost_visited[neigh_node] = new_cost
                visited[neigh_node] = cur_node

    # draw path
    path_head, path_segment = cur_node, cur_node
    while path_segment:
        pg.draw.circle(sc, pg.Color('brown'), *get_circle(*path_segment))
        path_segment = visited[path_segment]
    pg.draw.circle(sc, pg.Color('blue'), *get_circle(*start))
    pg.draw.circle(sc, pg.Color('magenta'), *get_circle(*path_head))
    # pygame necessary lines
    [exit() for event in pg.event.get() if event.type == pg.QUIT]
    pg.display.flip()
    clock.tick(7)