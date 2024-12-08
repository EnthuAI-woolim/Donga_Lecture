def read_bst_input(file_name):
    # 파일을 열어서 읽기 모드로 설정
    with open(file_name, 'r') as file:
        # 파일의 모든 라인을 읽어서 각 줄에 있는 숫자들을 차례대로 하나의 리스트에 저장
        data = [int(num) for line in file for num in line.split()]
    return data


class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


def insert(root, key):
    if root is None:
        return Node(key)

    if key < root.key:
        root.left = insert(root.left, key)
    else:
        root.right = insert(root.right, key)

    return root


def search(root, key, path="R"):
    # Base case: If the root is None or we found the key
    if root is None:
        return None

    # Check if the current node is the key we're searching for
    if root.key == key:
        return path

    # If the key is smaller, go left and append '0' to path
    if key < root.key:
        return search(root.left, key, path + "0")

    # If the key is greater, go right and append '1' to path
    return search(root.right, key, path + "1")


def delete(root, key):
    if root is None:
        return root

    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # 노드를 삭제해야 할 경우
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left

        # 두 자식이 있는 경우, 가장 작은 값으로 교체
        root.key = min_value(root.right)
        root.right = delete(root.right, root.key)

    return root


def min_value(node):
    current = node
    while current.left is not None:
        current = current.left
    return current.key


if __name__ == "__main__":
    data = read_bst_input("bst_input.txt")
    index = 0
    test_cases = data[index]
    index += 1

    # 파일에 출력하기 위해 open()을 사용하여 파일을 엽니다.
    with open("bst_output.txt", "w") as output_file:
        for _ in range(test_cases):
            # 삽입할 데이터 갯수
            insert_count = data[index]
            index += 1

            # 삽입할 데이터들
            insert_data = data[index:index + insert_count]
            index += insert_count

            # 첫번째 검색할 데이터 갯수
            search_count1 = data[index]
            index += 1

            # 첫번째 검색할 데이터들
            search_data1 = data[index:index + search_count1]
            index += search_count1

            # 삭제할 데이터 갯수
            delete_count = data[index]
            index += 1

            # 삭제할 데이터들
            delete_data = data[index:index + delete_count]
            index += delete_count

            # 두번째 검색할 데이터 갯수
            search_count2 = data[index]
            index += 1

            # 두번째 검색할 데이터들
            search_data2 = data[index:index + search_count2]
            index += search_count2

            # 이진 탐색 트리 만들기
            root = None

            # 삽입
            for key in insert_data:
                root = insert(root, key)

            # 검색1
            for key in search_data1:
                path = search(root, key)
                if path is None:
                    output_file.write(f"Search {key}: Not Found\n")
                else:
                    output_file.write(f"{path}\n")

            # 삭제
            for key in delete_data:
                root = delete(root, key)

            # 검색2
            for key in search_data2:
                path = search(root, key)
                if path is None:
                    output_file.write(f"Search {key}: Not Found\n")
                else:
                    output_file.write(f"{path}\n")
