def read_input_file(filename):
    with open(filename, 'r') as file:
        datas = []
        # 첫 줄에서 테스트 케이스 수를 읽어옴
        n_cases = int(file.readline().strip())

        # 각 테스트 케이스에 대해 반복
        for _ in range(n_cases):
            # 각 케이스의 레코드 수를 읽어옴 (현재는 사용하지 않지만 기록을 위해 읽음)
            n_records = int(file.readline().strip())

            # 레코드 줄을 읽고 공백을 기준으로 숫자로 변환하여 리스트에 저장
            records = [int(x) for x in file.readline().strip().split()]
            datas.append([n_records, records])

    return datas

if __name__=="__main__":
    data_sets = read_input_file('replacement_input.txt')
    m = 5
    result = []

    for case in data_sets: # 하나의 테스트 세트: 전체의 테스트 세트
        # freeze : is freeze, buf : key value
        freeze = [0] * m
        # if freeze[].count(2) == 1:
        #     print("!")

        buf = [0] * m
        runs = []
        run = []

        # print(case[1])

        for i in range(m): # 처음 5개의 key를 buf에 넣기
            buf[i] = case[1][i]

        for i in range(m, case[0]): # 하나의 key :하나의 테스트세트
            min_k = float('inf')
            min_idx = 0
            if freeze.count(1) == m: # 만약 전부 freeze 됐다면, runs에 현재 run추가하고 run 초기화
                runs.append(run)
                run = []
                freeze = [0] * m


            for j in range(m): # buf의 key값중 freeze 안되면서, 최소 값 찾기
                if freeze[j] == 0 and buf[j] <= min_k:
                    min_k = buf[j]
                    min_idx = j
            # -> buf에서 최솟값의 인덱스 찾음

            run.append(min_k) # run에 최솟값 넣기
            buf[min_idx] = case[1][i] # 다음 key값을 최솟값이 있었던 자리에 넣기
            if buf[min_idx] < min_k: # 만약 다음 key값이 최솟값보다 작다면 freeze == 1
                freeze[min_idx] = 1

        while freeze.count(2) != m:
            if freeze.count(0) == 0:
                runs.append(run)
                run =[]

            if freeze.count(0):
                min_k = float('inf')
                min_idx = 0
                for i in range(m):
                    if freeze[i] == 0 and buf[i] <= min_k:
                        min_k = buf[i]
                        min_idx = i

                run.append(min_k)
                freeze[min_idx] = 2
                if freeze.count(0) == 0:
                    runs.append(run)
                    run = []

            elif freeze.count(1):
                min_k = float('inf')
                min_idx = 0
                for i in range(m):
                    if freeze[i] == 1 and buf[i] <= min_k:
                        min_k = buf[i]
                        min_idx = i

                run.append(min_k)
                freeze[min_idx] = 2
                if freeze.count(1) == 0:
                    runs.append(run)
                    run = []
        result.append(runs)

    for i in range(2):
        print(result[i])




# # 데이터 출력 (테스트용)
# for index, case in enumerate(data):
#     print(f"Test Case {index + 1}: {case}")
