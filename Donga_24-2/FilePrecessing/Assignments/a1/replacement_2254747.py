def read_file(filename):
    with open(filename, 'r') as file:

        n_cases = int(file.readline().strip())

        result = [
            (int(file.readline().strip()), list(map(int, file.readline().strip().split())))
            for _ in range(n_cases)
        ]

        return result

def write_file(filename, results):
    with open(filename, 'w') as file:
        for case in results:
            file.write(f"{case[0]}\n")

            for run in case[1]:
                file.write(" ".join(str(i) for i in run))
                file.write("\n")


def init_buf_freeze(m, case):
    return case[:m], [0] * m


def find_min(buf, freeze, m, state):
    min_idx = min(
        (i for i in range(m) if freeze[i] == state),
        key=lambda i: buf[i]
    )
    return buf[min_idx], min_idx


def add_run(runs, run):
    runs.append(run)
    return [], len(runs)


def update_freeze_and_run(buf, freeze, run, m, state):
    min_k, min_idx = find_min(buf, freeze, m, state)
    run.append(min_k)
    freeze[min_idx] = 2
    return freeze, run


def replacement_selection(data_sets, m):
    result = []

    for n, case in data_sets:
        buf, freeze = init_buf_freeze(m, case)
        runs, run = [], []
        n_rums = 0

        for i in range(m, n):
            if freeze.count(1) == m:
                freeze = [0] * m
                run, n_rums = add_run(runs, run)

            min_k, min_idx = find_min(buf, freeze, m, 0)
            run.append(min_k)
            buf[min_idx] = case[i]
            if buf[min_idx] < min_k:
                freeze[min_idx] = 1

        if freeze.count(0) == 0:
            run, n_rums = add_run(runs, run)

        while freeze.count(2) != m:
            state = 0 if freeze.count(0) else 1
            freeze, run = update_freeze_and_run(buf, freeze, run, m, state)
            if freeze.count(state) == 0:
                run, n_rums = add_run(runs, run)

        result.append([n_rums, runs])

    return result

# python 3.11
if __name__ == "__main__":
    m = 5
    data_sets = read_file('replacement_input.txt')
    result = replacement_selection(data_sets, m)
    write_file('replacement_output.txt', result)

    # 결과 출력
    print(f"\n총 {len(result)}개")
    for i, case in enumerate(result, start=1):
        print(f"\nCase {i}: \n{case[0]}개")
        for run in case[1]:
            print(f"Run - {run}")

