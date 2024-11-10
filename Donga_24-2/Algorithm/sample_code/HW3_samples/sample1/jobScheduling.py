jobs = [['t1', 7, 8], ['t2', 3, 7], ['t3', 1, 5], ['t4', 5, 9], ['t5', 0, 2], ['t6', 6, 8], ['t7', 1, 6]]  #작업을 담은 리스트
machines = []  #기계들의 작업을 담을 리스트
jobs.sort(key=lambda x: x[1])  #작업들을 시작시간을 기준으로 오름차순 정렬

for job in jobs:  #리스트 안에 들어있는 모든 작업들에 대해 반복
    success = False  #기존에 있던 기계에 작업을 추가했다면 success는 True가 된다.
    start = job[1]  #현재 선택된 작업의 시작시간
    end = job[2]  #현재 선택된 작업의 종료시간

    for machine in machines:  #기계들에 들어있는 작업들을 확인한다.
        if machine[-1][2] <= start:
            machine.append(job)
            success = True
            break

    if not success: #배정할 기계가 없으면 새로운 기계에 배정한다.
        machines.append([job])

totalTime = max(job[2] for job in jobs)

print('  time      ', end='')
for i in range(totalTime + 1):
    print(f'{i:<8}', end='')  #시간들 출력 (칸 8칸으로 맞춤)
print()

index = len(machines)
machines.reverse()
for machine in machines:
    print(f"machine {index:<2}  ", end='')  #기계 번호도 8칸으로 맞춤
    length = len(machine)
    for i in range(length):
        if i == 0 and machine[0][1] != 0:
            for _ in range(machine[0][1]):
                print('        ', end='')  #8칸 공백

        if i > 0 and machine[i][1] > machine[i-1][2]:
            for _ in range(machine[i][1] - machine[i-1][2]):
                print('        ', end='')  #8칸 공백

        for _ in range(machine[i][1], machine[i][2]):
            print(f'{machine[i][0]:<8}', end='')  #작업 이름도 8칸으로 맞춤
    print()
    index -= 1
