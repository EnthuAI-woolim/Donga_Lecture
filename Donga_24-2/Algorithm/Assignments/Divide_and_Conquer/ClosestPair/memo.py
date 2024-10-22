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


def dis_of_strip(l_points, r_points, min_dis):
    # 중간 영역의 점들을 비교했을때, 더 작은 거리가 나오는 경우는
    # 1. left의 한 점, rigth의 한 점
    # 2. 값이 똑같은 두개의 점(이 경우 l_points, r_points 각각에 값이 있어야됨)
    for l in l_points:
        for r in r_points:
            # points는 y값을 기준으로 정렬되어 있기 때문에
            # y값의 차이가 min_dis보다 커지는 시점에서 break하여 points 전부 연산하는 것을 방지
            if (l[1] - r[1]) ** 2 >= min_dis: break 
            cur_dis = dis_sqr(l, r)
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

    l_strip, r_strip = [], []
    for point in sorted_on_y:
        if mid_point[0] - min_dis < point[0] <= mid_point[0]: l_strip.append(point)
        if mid_point[0] <= point[0] < mid_point[0] + min_dis: r_strip.append(point)
    strip_dis = dis_of_strip(l_strip, r_strip, min_dis)

    return min(min_dis, strip_dis)


def closestPair(points):
    sorted_on_x = sort_by_column(points, column=0)  # x값 기준으로 정렬된 점들
    sorted_on_y = sort_by_column(points, column=1)  # y값 기준으로 정렬된 점들
    closest_dis_sqr = closestPair_sqr(sorted_on_x, sorted_on_y) # 제곱된 거리가 값이 return됨

    return closest_dis_sqr ** 0.5

def read_pairs(filename):
    with open(filename, 'r') as file:
        data = file.read()
    points = data.split()

    return [(int(points[i]),int(points[i+1])) for i in range(0, len(points)-1, 2)]

def main():
    points = read_pairs('../input_closest_pair.txt')
    start = time.time()
    dis = closestPair(points)
    end = time.time()
    exec_time = round((end-start)*1000, 6)
    print("가장 가까운 두 점 사이의 거리:", dis)
    print("running time: {}ms".format(exec_time))

if __name__ == "__main__":
    main()
