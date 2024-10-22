import math
import time
import copy

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def bruteForce(P, n):
    min_val = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if dist(P[i], P[j]) < min_val:
                min_val = dist(P[i], P[j])
    return min_val

def stripClosest(strip, size, d):
    min_val = d
    for i in range(size):
        j = i + 1
        while j < size and (strip[j].y - strip[i].y) < min_val:
            min_val = dist(strip[i], strip[j])
            j += 1
    return min_val

def closestUtil(P, Q, n):
    if n <= 3:
        return bruteForce(P, n)

    mid = n // 2
    midPoint = P[mid]

    Pl = P[:mid]
    Pr = P[mid:]

    dl = closestUtil(Pl, Q, mid)
    dr = closestUtil(Pr, Q, n - mid)

    d = min(dl, dr)

    stripP = []
    stripQ = []
    lr = Pl + Pr
    for i in range(n):
        if abs(lr[i].x - midPoint.x) < d:
            stripP.append(lr[i])
        if abs(Q[i].x - midPoint.x) < d:
            stripQ.append(Q[i])

    stripP.sort(key=lambda point: point.y)
    min_a = min(d, stripClosest(stripP, len(stripP), d))
    min_b = min(d, stripClosest(stripQ, len(stripQ), d))

    return min(min_a, min_b)

def closest(P, n):
    P.sort(key=lambda point: point.x)
    Q = copy.deepcopy(P)
    Q.sort(key=lambda point: point.y)

    return closestUtil(P, Q, n)

def read_input(file_name):
    points = []
    with open(file_name, 'r') as file:
        seen_points = set() 
        for line in file:
            x, y = map(int, line.split())
            point = (x, y)
            if point in seen_points:
                print(f"Duplicate point found: {point}")
            else:
                seen_points.add(point)
                points.append(Point(x, y))
    return points

if __name__ == "__main__":
    input_file = 'input_closest_pair.txt'

    P = read_input(input_file)
    n = len(P)

    start_time = time.time()

    result = closest(P, n)

    end_time = time.time()

    print("최소거리", result)
    print("Running time: {:.6f} seconds".format(end_time - start_time))
