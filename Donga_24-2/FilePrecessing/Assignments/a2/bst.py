class BinarySearchTree:
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def insert(self, key):
        def _insert(node, key):
            if node is None:
                return BinarySearchTree.Node(key)
            if key < node.key:
                node.left = _insert(node.left, key)
            else:
                node.right = _insert(node.right, key)
            return node

        self.root = _insert(self.root, key)

    def search(self, key):
        def _search(node, key, path="R"):
            if node is None:
                return None
            if node.key == key:
                return path
            if key < node.key:
                return _search(node.left, key, path + "0")
            return _search(node.right, key, path + "1")

        return _search(self.root, key)

    def delete(self, key):
        def _delete(node, key):
            if node is None:
                return node
            if key < node.key:
                node.left = _delete(node.left, key)
            elif key > node.key:
                node.right = _delete(node.right, key)
            else:
                if node.left is None:
                    return node.right
                elif node.right is None:
                    return node.left
                min_val = self._min_value(node.right)
                node.key = min_val
                node.right = _delete(node.right, min_val)
            return node

        self.root = _delete(self.root, key)

    def _min_value(self, node):
        while node.left is not None:
            node = node.left
        return node.key


if __name__ == "__main__":
    with open('bst_input.txt', 'r') as file:
        data = list(map(int, file.read().split()))


    test_cases = data[0]
    index = 1
    results = []

    for _ in range(test_cases):
        bst = BinarySearchTree()

        # 삽입
        insert_count = data[index]
        index += 1
        insert_data = data[index:index + insert_count]
        index += insert_count
        for key in insert_data:
            bst.insert(key)

        # 검색1
        search_count1 = data[index]
        index += 1
        search_data1 = data[index:index + search_count1]
        index += search_count1
        for key in search_data1:
            path = bst.search(key)
            if path is None:
                results.append(f"Not Found: {key}")
            else:
                results.append(path)

        # 삭제
        delete_count = data[index]
        index += 1
        delete_data = data[index:index + delete_count]
        index += delete_count
        for key in delete_data:
            bst.delete(key)

        # 검색2
        search_count2 = data[index]
        index += 1
        search_data2 = data[index:index + search_count2]
        index += search_count2
        for key in search_data2:
            path = bst.search(key)
            if path is None:
                results.append(f"Not Found: {key}")
            else:
                results.append(path)

    print("\n".join(results) + "\n")

    # 결과를 파일에 저장
    with open('bst_output.txt', "w") as output_file:
        output_file.write("\n".join(results) + "\n")

