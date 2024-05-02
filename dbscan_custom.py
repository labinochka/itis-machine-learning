import pygame
import numpy as np


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_neighbors(points, point, epsilon):
    neighbors = []
    for other_point in points:
        if other_point != point and distance(point, other_point) <= epsilon:
            neighbors.append(other_point)
    return neighbors


def dbscan(points, epsilon, min_points):
    clusters = {}
    cluster_id = 0

    for point in points:
        if point in clusters:
            continue

        neighbors = find_neighbors(points, point, epsilon)

        if len(neighbors) < min_points:
            clusters[point] = -1
            continue

        clusters[point] = cluster_id

        queue = neighbors.copy()
        while queue:
            current_point = queue.pop(0)
            if current_point in clusters and clusters[current_point] != -1:
                continue
            clusters[current_point] = cluster_id
            current_neighbors = find_neighbors(points, current_point, epsilon)
            if len(current_neighbors) >= min_points:
                queue.extend(current_neighbors)

        cluster_id += 1

    return clusters


if __name__ == '__main__':
    points = []
    r = 10
    min_pts, eps = 3, 3 * r
    colors = ['green', 'yellow', 'blue', 'purple', 'pink', 'gray', 'orange', 'red']
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    screen.fill('white')
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    point = pygame.mouse.get_pos()
                    points.append(point)
                    pygame.draw.circle(screen, 'black', point, r)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    clusters = dbscan(points, eps, min_pts)
                    for point in points:
                        cluster_id = clusters[point]
                        color = colors[cluster_id];
                        pygame.draw.circle(screen, color, point, r)

        pygame.display.flip()
    pygame.quit()
