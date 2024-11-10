

'''
알고리즘 순서
1) 작업 클래스 생성 
2) 머신 리스트 생성 
3) 작업 리스트 생성
4) 작업을 정렬
5) 작업 리스트를 순회하면서 작업을 배치
작업을 수행할 머신이 있으면 그곳에 작업을 배치하고, 없으면 새로운 머신을 생성하고 작업을 배치
6) 머신 리스트를 역순으로 출력
'''

class Job:
    def __init__(self, name, start_time , end_time):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time

# 0) input
# t1=[7, 8], t2=[3, 7], t3=[1, 5], t4=[5, 9], t5=[0, 2], t6=[6, 8], t7=[1, 6]
job_List = []  
jobs = [
    ('t1', 7, 8), ('t2', 3, 7), ('t3', 1, 5), 
    ('t4', 5, 9), ('t5', 0, 2), ('t6', 6, 8), ('t7', 1, 6)
]

# 1) job_List 초기화
job_List = [Job(name, start_time, end_time) for name, start_time, end_time in jobs]

machines = [[0]*9]  # 10개의 기계, 각 기계는 10개의 시간 슬롯을 가짐  ''' {{}} 몇번 job인지 넣는 곳입니다. {{1,1,1,1}, {0,0,0,0,2,2,2}} 이면, 0번 머신은 0부터 3까지 일을 하는거고, 1번 머신은 4번부터 6번까지 일하는겁니다.'''

# 2) 정렬 
job_List.sort(key=lambda x:x.start_time) # start_time 을 기준으로 정렬합니다. 오름차순으로 정렬합니다.

# job list 순회하면서
while job_List:
    first_job = job_List.pop(0)
    machine_OK = False
    machine_idx = -1
    # 작업할 머신이 있는지 일차적으로 확인한다.
    for m_idx, machine in enumerate(machines):
        # 일 할 수 있는 머신이 있으면 True , 아니면 False
        # job이 머신을 순회하면서 자리가 있는지 확인한다.
        if machine[first_job.start_time] == 0:
            machine_OK = True
            machine_idx = m_idx
            break

    # 자리가 있으면) 그곳에 작업배치
    if machine_OK:
        for i in range(first_job.start_time,first_job.end_time):
            machines[machine_idx][i] = first_job.name
    # 자리가 없으면) 자리를 만들고, 생성합니다. 
    else:
        new_machine = [0] * 9
        for i in range(first_job.start_time,first_job.end_time):
            new_machine[i] = first_job.name
        machines.append(new_machine)            

# 역순으로 출력
m_number = 3 
print("          ",end="")
for i in range(9):
    print(i,end="     ")
print()
for m in reversed(machines):
    print("machine",m_number,m)
    m_number -= 1




            
            

    
