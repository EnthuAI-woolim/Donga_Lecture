import time

def dis_sqr(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def sort_by_column(arr, column=0):
    return sorted(arr, key=lambda x: x[column])

def dis_of_closestPair(points, counts):
    min_dis = float("inf")
    for i in range(counts - 1):
        for j in range(i + 1, counts):
            cur_dis = dis_sqr(points[i], points[j])
            min_dis = min(min_dis, cur_dis)
    return min_dis


def dis_of_strip(points, min_dis):
    counts = len(points)

    for i in range(counts):
        for j in range(i + 1, counts):
            cur_dis = dis_sqr(points[i], points[j])
            min_dis = min(min_dis, cur_dis)

    return min_dis

def closestPair_sqr(sorted_on_x, sorted_on_y):
    counts = len(sorted_on_x)

    if counts <= 3:
        return dis_of_closestPair(sorted_on_x, counts)

    mid = counts // 2
    mid_point = sorted_on_x[mid]

    left_y = [point for point in sorted_on_y if point[0] <= mid_point[0]]
    right_y = [point for point in sorted_on_y if point[0] > mid_point[0]]
    left_dist = closestPair_sqr(sorted_on_x[:mid], left_y)
    right_dist = closestPair_sqr(sorted_on_x[mid:], right_y)
    min_dis = min(left_dist, right_dist)

    cross_strip = [point for point in sorted_on_y if abs(point[0] - mid_point[0]) < min_dis]
    strip_dis = dis_of_strip(cross_strip, min_dis)

    return min(min_dis, strip_dis)


def closetPair(points):
    sorted_on_x = sort_by_column(points, column=0)
    sorted_on_y = sort_by_column(points, column=1)
    closest_dis_sqr = closestPair_sqr(sorted_on_x, sorted_on_y)

    return closest_dis_sqr ** 0.5

def read_pairs():
    with open('input_closest_pair.txt', 'r') as file:
        data = file.read()
    points = data.split()

    return [(int(points[i]),int(points[i+1])) for i in range(0, len(points)-1, 2)]

def main():
    points = read_pairs()
    start = time.time()
    dis = closetPair(points)
    end = time.time()
    exec_time = round((end-start)*1000, 6)
    print("가장 가까운 두 점 사이의 거리:", dis)
    print("running time: {}ms".format(exec_time))

if __name__ == "__main__":
    main()
