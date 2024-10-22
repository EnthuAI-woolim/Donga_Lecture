import math
import time

# x y로 이루어진 Point 클래스
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 점 간의 거리 측정 함수
def distance(p1: Point, p2: Point):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def stripClosest(strip, size, d, d_pair):
    min_val = d
    min_pair = d
    for i in range(size):
        j = i+1
        while j < size and distance(strip[i], strip[j]) < min_val: #abs(strip[j].y - strip[i].y) < min_val:
            min_val = distance(strip[i], strip[j])
            min_pair = (strip[i], strip[j])
            j += 1
    
    return (min_val, min_pair)

def closestUnit(arr1, arr2, n):
    if n <= 3: # 좌표 개수가 3보다 작거나 같은 경우
        min_val = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                if distance(arr1[i], arr1[j]) < min_val:
                    min_val = distance(arr1[i], arr1[j])
                    min_pair = (arr1[i], arr1[j])

        return (min_val, min_pair) # 최근접 쌍 반환
    
    # 양쪽으로 분할
    mid = n // 2
    midPoint = arr1[mid]
    l1 = arr1[:mid]
    r1 = arr1[mid:]

    dl, l_pair = closestUnit(l1, arr2, mid) # 왼쪽에 대한 최근접 점의 쌍 반환
    dr, r_pair = closestUnit(r1, arr2, n - mid) # 오른쪽에 대한 최근접 점의 쌍 반환

    # 양쪽 최근접 점 거리 중 더 짧은 것을 선택
    if dl < dr:
        d = dl
        d_pair = l_pair
    else:
        d = dr
        d_pair = r_pair

    # 중간 영역에서의 최근접 점 쌍 찾기
    stripArr1 = []
    stripArr2 = []
    lr = l1 + r1

    # 양쪽에서 중간 영역에 포함되는 점을 리스트에 포함
    for i in range(n):
        if abs(lr[i].x - midPoint.x) < d:
            stripArr1.append(lr[i])
        if abs(arr2[i].x - midPoint.x) < d:
            stripArr2.append(arr2[i])

    # 각각의 중간 영역에서 최근접 점의 쌍을 반환받음
    stripArr1.sort(key=lambda p:p.y)
    strip1, strip1_pair = stripClosest(stripArr1, len(stripArr1), d, d_pair)
    strip2, strip2_pair = stripClosest(stripArr2, len(stripArr2), d, d_pair)

    # 중간 영역에서 찾은 최근접 점의 쌍을 현재 최근접 점의 쌍과 비교
    if strip1 < d:
        min_a = strip1
        min_a_pair = strip1_pair
    else:
        min_a = d
        min_a_pair = d_pair
    
    if strip2 < d:
        min_b = strip2
        min_b_pair = strip2_pair
    else:
        min_b = d
        min_b_pair = d_pair
    
    # 최종적인 최근접 점의 쌍 반환
    if min_a < min_b:
        return (min_a, min_a_pair)
    else:
        return (min_b, min_b_pair)

def closestPair(arr, n):
    import copy
    distances = []
    pairs = []
    while True: # 가장 짧은 거리의 쌍을 모두 찾을 때까지 반복
        arr.sort(key=lambda p:p.x)
        arr2 = copy.deepcopy(arr)
        arr2.sort(key=lambda p:p.y)

        distance, pair = closestUnit(arr, arr2, n)
        
        # 이전의 최근접 점 쌍의 거리보다 큰 거리가 발생한 경우 종료
        if len(distances) > 1 and distances[-1] != distance:
            break
        distances.append(distance)
        pairs.append(pair)

        # 발견된 쌍은 리스트에서 삭제
        arr = [p for p in arr if not ((p.x == pair[0].x and p.y == pair[0].y) or (p.x == pair[1].x and p.y == pair[1].y))]

        n -= 2
    
    return distances, pairs

def main():
    # 파일 입력 받기
    arr = []
    with open("/workspace/HW2/ClosestPair/input_closest_pair.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            x, y = line.split()[0], line.split()[1]
            x = int(x)
            y = int(y)
            arr.append(Point(x, y))

    # 시간 측정
    start = time.time()
    distances, pairs = (closestPair(arr, len(arr)))
    end = time.time()

    # 가장 짧은 거리의 점의 짝과 거리를 출력
    for dist, pair in zip(distances, pairs):
        print(f'짝: ({pair[0].x}, {pair[0].y}), ({pair[1].x}, {pair[1].y})')
        print(f'거리: {dist}')

    print(f"Running time: {end - start:.5f}sec")

if __name__ == "__main__":
    main()