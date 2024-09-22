"""
가장 가까운 두 점 사이의 거리를 찾는 알고리즘.
알고리즘은 분할 정복 방식을 사용하여 주어진 n개의 점들에서
가장 가까운 두 점 사이의 유클리드 거리의 제곱을 계산합니다.

시간 복잡도: O(n * log n)
"""

def euclidean_distance_sqr(point1, point2):
    """
    두 점 사이의 유클리드 거리의 제곱을 반환.

    >>> euclidean_distance_sqr([1,2],[2,4])
    5
    """
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def column_based_sort(array, column=0):
    """
    배열을 주어진 열(column)을 기준으로 정렬.

    >>> column_based_sort([(5, 1), (4, 2), (3, 0)], 1)
    [(3, 0), (5, 1), (4, 2)]
    """
    return sorted(array, key=lambda x: x[column])


def dis_between_closest_pair(points, points_counts, min_dis=float("inf")):
    """
    브루트포스 방식으로 가장 가까운 두 점 사이의 거리 계산.

    >>> dis_between_closest_pair([[1,2],[2,4],[5,7],[8,9],[11,0]],5)
    5
    """
    for i in range(points_counts - 1):
        for j in range(i + 1, points_counts):
            current_dis = euclidean_distance_sqr(points[i], points[j])
            min_dis = min(min_dis, current_dis)
    return min_dis


def dis_between_closest_in_strip(points, points_counts, min_dis=float("inf")):
    """
    스트립 내에서 가장 가까운 두 점 사이의 거리 계산.

    >>> dis_between_closest_in_strip([[1,2],[2,4],[5,7],[8,9],[11,0]],5)
    85
    """
    for i in range(min(6, points_counts - 1), points_counts):
        for j in range(max(0, i - 6), i):
            current_dis = euclidean_distance_sqr(points[i], points[j])
            min_dis = min(min_dis, current_dis)
    return min_dis


def closest_pair_of_points_sqr(points_sorted_on_x, points_sorted_on_y, points_counts):
    """
    분할 정복을 사용하여 가장 가까운 두 점 사이의 거리를 찾는 함수.

    >>> closest_pair_of_points_sqr([(1, 2), (3, 4)], [(5, 6), (7, 8)], 2)
    8
    """
    if points_counts <= 3:
        return dis_between_closest_pair(points_sorted_on_x, points_counts)

    mid = points_counts // 2
    closest_in_left = closest_pair_of_points_sqr(
        points_sorted_on_x[:mid], points_sorted_on_y[:mid], mid
    )
    closest_in_right = closest_pair_of_points_sqr(
        points_sorted_on_x[mid:], points_sorted_on_y[mid:], points_counts - mid
    )
    closest_pair_dis = min(closest_in_left, closest_in_right)

    cross_strip = [
        point for point in points_sorted_on_y
        if abs(point[0] - points_sorted_on_x[mid][0]) < closest_pair_dis
    ]

    closest_in_strip = dis_between_closest_in_strip(
        cross_strip, len(cross_strip), closest_pair_dis
    )

    return min(closest_pair_dis, closest_in_strip)


def closest_pair_of_points(points, points_counts):
    """
    주어진 점들 중 가장 가까운 두 점 사이의 유클리드 거리를 찾는 함수.

    >>> closest_pair_of_points([(2, 3), (12, 30)], len([(2, 3), (12, 30)]))
    28.792360097775937
    """
    points_sorted_on_x = column_based_sort(points, column=0)
    points_sorted_on_y = column_based_sort(points, column=1)
    return closest_pair_of_points_sqr(
        points_sorted_on_x, points_sorted_on_y, points_counts
    ) ** 0.5


if __name__ == "__main__":
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    print("가장 가까운 두 점 사이의 거리:", closest_pair_of_points(points, len(points)))
