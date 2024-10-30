def allocate_tasks(tasks):
    tasks.sort(key=lambda x: x[1])  # 작업 시작 시간을 기준으로 정렬
    machines = [[tasks[0]]]
    end_times = [tasks[0][2]]
    num_machines = 0

    for i in range(1, len(tasks)):
        allocated = False
        for j in range(num_machines + 1):
            if tasks[i][1] >= end_times[j]:  # 새로운 작업의 시작시간이 현재 작업의 끝시간보다 같거나 클 경우
                machines[j].append(tasks[i])
                end_times[j] = tasks[i][2]
                allocated = True
                break

        if not allocated:  # 배정 불가시 새로운 기계에 배정
            num_machines += 1
            machines.append([tasks[i]])
            end_times.append(tasks[i][2])

    return machines

def print_schedule(machines):
    max_time = max(t[2] for t in tasks)

    print("\ntime     ", end=" ")
    for t in range(max_time):
        print(f"{t:>3}", end=" ")
    print()

    for idx, schedule in enumerate(machines, start=1):
        print(f"Machine{idx:<3}", end=" ")

        timeline = ["  "] * max_time
        for name, start, end in schedule:
            for t in range(start, end):
                timeline[t] = name

        print("  ".join(timeline))
    print()

if __name__ == "__main__":
    tasks = [['t1', 7, 8], ['t2', 3, 7], ['t3', 1, 5], ['t4', 5, 9], ['t5', 0, 2], ['t6', 6, 8], ['t7', 1, 6]]
    machines = allocate_tasks(tasks)
    print_schedule(machines)