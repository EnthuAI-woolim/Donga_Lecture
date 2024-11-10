# 작업 리스트 초기화 (각 작업은 [시작 시간, 종료 시간]으로 표현)
tasks = {
    't1': [7, 8],
    't2': [3, 7],
    't3': [1, 5],
    't4': [5, 9],
    't5': [0, 2],
    't6': [6, 8],
    't7': [1, 6]
}

# 시작 시간으로 정렬하고, 시작 시간이 같으면 작업 시간이 긴 순서대로 정렬
sorted_tasks = sorted(tasks.items(), key=lambda x: (x[1][0], -(x[1][1] - x[1][0])))

# 기계 리스트 초기화
machines = []

# 각 작업을 기계에 배정
for task, (start, end) in sorted_tasks:
    assigned = False
    # 현재 작업을 수행할 수 있는 기계 찾기
    for machine in machines:
        # 기계의 마지막 작업 종료 시간이 현재 작업의 시작 시간보다 작거나 같아야 함
        if machine[-1][2] <= start:
            machine.append((task, start, end))
            assigned = True
            break
    # 배정할 수 있는 기계가 없으면 새 기계 추가
    if not assigned:
        machines.append([(task, start, end)])

# 기계를 역순으로 정렬하여 Machine 1이 가장 아래에 오도록 설정
machines = machines[::-1]

# 최대 시간 값 계산
max_time = max(end for task, (start, end) in tasks.items())

# 결과 출력
print("time	", end="\t")
for t in range(max_time):
    print(t, end="\t")
print()

# 각 기계에 배정된 작업을 시간대별로 출력
for i, machine in enumerate(machines, start=1):
    row = [""] * (max_time + 1)
    for task, start, end in machine:
        for t in range(start, end):
            row[t] = task
    print(f"Machine {len(machines) - i + 1}", end="\t")
    for slot in row:
        print(slot if slot else "", end="\t")
    print()

