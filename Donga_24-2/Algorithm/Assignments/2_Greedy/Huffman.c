#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TREE_HT 100
#define MAX_CHARACTERS 256 

// 허프만 트리 노드 구조체
struct MinHeapNode {
    char data;
    unsigned freq;
    struct MinHeapNode *left, *right;
};

// 최소 힙 구조체
struct MinHeap {
    unsigned size;
    unsigned capacity;
    struct MinHeapNode** array;
};

// 새 노드 생성 함수
struct MinHeapNode* newNode(char data, unsigned freq) {
    struct MinHeapNode* temp = (struct MinHeapNode*)malloc(sizeof(struct MinHeapNode));
    temp->left = temp->right = NULL;
    temp->data = data;
    temp->freq = freq;
    return temp;
}

// 최소 힙 생성 함수
struct MinHeap* createMinHeap(unsigned capacity) {
    struct MinHeap* minHeap = (struct MinHeap*)malloc(sizeof(struct MinHeap));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array = (struct MinHeapNode**)malloc(minHeap->capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}

// 노드 교환 함수
void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b) {
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

// 최소 힙 재정렬 함수
void minHeapify(struct MinHeap* minHeap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < minHeap->size && minHeap->array[left]->freq < minHeap->array[smallest]->freq)
        smallest = left;

    if (right < minHeap->size && minHeap->array[right]->freq < minHeap->array[smallest]->freq)
        smallest = right;

    if (smallest != idx) {
        swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}

// 힙 크기가 1인지 확인
int isSizeOne(struct MinHeap* minHeap) {
    return (minHeap->size == 1);
}

// 최소 값 노드 추출
struct MinHeapNode* extractMin(struct MinHeap* minHeap) {
    struct MinHeapNode* temp = minHeap->array[0];
    minHeap->array[0] = minHeap->array[minHeap->size - 1];
    --minHeap->size;
    minHeapify(minHeap, 0);
    return temp;
}

// 최소 힙에 새 노드 삽입
void insertMinHeap(struct MinHeap* minHeap, struct MinHeapNode* minHeapNode) {
    ++minHeap->size;
    int i = minHeap->size - 1;

    while (i && minHeapNode->freq < minHeap->array[(i - 1) / 2]->freq) {
        minHeap->array[i] = minHeap->array[(i - 1) / 2];
        i = (i - 1) / 2;
    }

    minHeap->array[i] = minHeapNode;
}

// 최소 힙 생성 함수
void buildMinHeap(struct MinHeap* minHeap) {
    int n = minHeap->size - 1;
    for (int i = (n - 1) / 2; i >= 0; --i)
        minHeapify(minHeap, i);
}

// 리프 노드인지 확인하는 함수
int isLeaf(struct MinHeapNode* root) {
    return !(root->left) && !(root->right);
}

// 문자의 빈도수 기준으로 최소 힙 생성
struct MinHeap* createAndBuildMinHeap(char data[], int freq[], int size) {
    struct MinHeap* minHeap = createMinHeap(size);
    for (int i = 0; i < size; ++i)
        minHeap->array[i] = newNode(data[i], freq[i]);
    minHeap->size = size;
    buildMinHeap(minHeap);
    return minHeap;
}

// 허프만 트리 생성 함수
struct MinHeapNode* buildHuffmanTree(char data[], int freq[], int size) {
    struct MinHeapNode *left, *right, *top;
    struct MinHeap* minHeap = createAndBuildMinHeap(data, freq, size);

    while (!isSizeOne(minHeap)) {
        left = extractMin(minHeap);
        right = extractMin(minHeap);
        top = newNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;
        insertMinHeap(minHeap, top);
    }

    return extractMin(minHeap);
}

// 각 문자의 허프만 코드 저장
void storeCodes(struct MinHeapNode* root, int arr[], int top, char huffmanCodes[MAX_CHARACTERS][MAX_TREE_HT]) {
    if (root->left) {
        arr[top] = 0;
        storeCodes(root->left, arr, top + 1, huffmanCodes);
    }
    if (root->right) {
        arr[top] = 1;
        storeCodes(root->right, arr, top + 1, huffmanCodes);
    }
    if (isLeaf(root)) {
        int i;
        for (i = 0; i < top; ++i)
            huffmanCodes[(int)root->data][i] = arr[i] + '0';  // 코드 저장
        huffmanCodes[(int)root->data][top] = '\0';            // 문자열 종료 문자
    }
}

// 허프만 트리 디코딩 함수
void decodeHuffman(struct MinHeapNode* root, const char* encodedStr) {
    struct MinHeapNode* current = root;
    for (int i = 0; encodedStr[i]; i++) {
        if (encodedStr[i] == '0')
            current = current->left;
        else
            current = current->right;

        // 리프 노드일 경우
        if (isLeaf(current)) {
            printf("%c", current->data);
            current = root;  // 다시 루트로 돌아감
        }
    }
    printf("\n\n");
}

// 문자 빈도 계산 함수
void calculateFrequency(const char* input, char data[], int freq[], int *size) {
    int count[MAX_CHARACTERS] = {0};
    for (int i = 0; input[i]; i++) count[(unsigned char)input[i]]++;
    for (int i = 0; i < MAX_CHARACTERS; i++) {
        if (count[i] > 0) {
            data[*size] = (char)i;
            freq[*size] = count[i];
            (*size)++;
        }
    }
}


int main() {
    FILE *file = fopen("Huffman_input.txt", "r");
    if (!file) {
        perror("파일을 열 수 없습니다");
        return 1;
    }

    char input[10000];
    fscanf(file, "%s", input);
    fclose(file);

    char data[MAX_CHARACTERS];
    int freq[MAX_CHARACTERS];
    int size = 0;

    calculateFrequency(input, data, freq, &size); // 각 문자의 빈도 수 계산
    struct MinHeapNode* root = buildHuffmanTree(data, freq, size); // 빈도수를 기준으로 트리 생성

    int arr[MAX_TREE_HT], top = 0;
    char huffmanCodes[MAX_CHARACTERS][MAX_TREE_HT] = {{0}};  // 허프만 코드를 저장하기 위한 배열

    storeCodes(root, arr, top, huffmanCodes); // 각 문자의 허프만 코드 저장

    // 허프만 코드 출력
    printf("\nHuffman Codes (");
    for (int i = 0; i < size; i++) {
        printf("'%c' = %s", data[i], huffmanCodes[(int)data[i]]);
        if (i != size-1) printf(", "); 
    }
    printf(")\n");

    // 문자열 디코딩
    const char* encodedStr = "10110010001110101010100";
    printf("\nDecoded result: ");
    decodeHuffman(root, encodedStr);

    return 0;
}