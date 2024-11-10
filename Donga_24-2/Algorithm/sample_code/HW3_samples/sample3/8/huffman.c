#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHAR_COUNT 4  // 'A', 'T', 'G', 'C' 4개의 문자만 고려
#define MAX_CODE_LENGTH 100

typedef struct Node {
    char character;
    int frequency;
    struct Node* left, * right;
} Node;

typedef struct {
    Node* data[CHAR_COUNT * 4];  // 큐 크기 설정
    int size;
} PriorityQueue;

// 우선순위 큐 초기화
void initQueue(PriorityQueue* pq) {
    pq->size = 0;
}

void enqueue(PriorityQueue* pq, Node* node) {
    int i = pq->size++;
    while (i && node->frequency < pq->data[(i - 1) / 2]->frequency) {
        pq->data[i] = pq->data[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    pq->data[i] = node;
}

Node* dequeue(PriorityQueue* pq) {
    Node* minNode = pq->data[0];
    pq->data[0] = pq->data[--pq->size];
    int i = 0;
    while (2 * i + 1 < pq->size) {
        int j = 2 * i + 1;
        if (j + 1 < pq->size && pq->data[j + 1]->frequency < pq->data[j]->frequency) j++;
        if (pq->data[i]->frequency <= pq->data[j]->frequency) break;
        Node* temp = pq->data[i];
        pq->data[i] = pq->data[j];
        pq->data[j] = temp;
        i = j;
    }
    return minNode;
}

// 새로운 노드 생성
Node* createNode(char character, int frequency) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->character = character;
    node->frequency = frequency;
    node->left = node->right = NULL;
    return node;
}

// Huffman 트리 생성
Node* buildHuffmanTree(int frequencies[]) {
    PriorityQueue pq;
    initQueue(&pq);

    // 각 문자에 대해 노드 생성 후 큐에 추가
    for (int i = 0; i < CHAR_COUNT; i++) {
        enqueue(&pq, createNode("ATGC"[i], frequencies[i]));
    }

    // 큐에 있는 노드 수 >= 2 (n-1번 반복)
    while (pq.size > 1) {
        Node* left = dequeue(&pq);
        Node* right = dequeue(&pq);
        Node* merged = createNode('\0', left->frequency + right->frequency);
        merged->left = left;
        merged->right = right;
        enqueue(&pq, merged);
    }

    return dequeue(&pq);  // Huffman 트리의 루트 반환
}

// Huffman 코드 생성
void generateHuffmanCodes(Node* root, char* code, int depth, char codes[128][MAX_CODE_LENGTH]) {
    if (!root) return;
    
    // 리프 노드에 도달한 경우
    if (root->character) {
        code[depth] = '\0';
        strcpy(codes[(int)root->character], code);
    } else {
        // 왼쪽 가지는 0, 오른쪽 가지는 1
        code[depth] = '0';
        generateHuffmanCodes(root->left, code, depth + 1, codes);
        code[depth] = '1';
        generateHuffmanCodes(root->right, code, depth + 1, codes);
    }
}

// 파일 읽고 빈도수 측정
void calculateFrequencies(const char* filename, int frequencies[]) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("파일 열기 오류");
        exit(1);
    }

    char ch;
    while ((ch = fgetc(file)) != EOF) {
        switch (ch) {
            case 'A':
                frequencies[0]++;
                break;
            case 'T':
                frequencies[1]++;
                break;
            case 'G':
                frequencies[2]++;
                break;
            case 'C':
                frequencies[3]++;
                break;
            default:
                break;
        }
    }
    fclose(file);

    /*// 빈도수 디버그 출력
    printf("빈도수 측정 결과:\n");
    printf("A: %d\n", frequencies[0]);
    printf("T: %d\n", frequencies[1]);
    printf("G: %d\n", frequencies[2]);
    printf("C: %d\n", frequencies[3]);*/
}

// 이진 코드 압축 해제
void decodeHuffmanCode(Node* root, const char* encodedStr) {
    Node* current = root;
    printf("압축 해제 결과: ");
    for (int i = 0; encodedStr[i] != '\0'; i++) {
        if (encodedStr[i] == '0') {
            current = current->left;
        } else {
            current = current->right;
        }

        // 리프 노드에 도달하면 해당 문자를 출력
        if (current->left == NULL && current->right == NULL) {
            printf("%c", current->character);
            current = root;
        }
    }
    printf("\n");
}

int main() {
    const char* filename = "Huffman_input.txt";
    int frequencies[CHAR_COUNT] = { 0 };  // 'A', 'T', 'G', 'C' 빈도수 초기화
    char codes[128][MAX_CODE_LENGTH] = { "" };  // ASCII 문자의 코드 저장
    const char* encodedStr = "10110010001110101010100";
	
     // 파일에서 빈도수 측정
    calculateFrequencies(filename, frequencies);
    
    // Huffman 트리 생성
    Node* root = buildHuffmanTree(frequencies);

    // Huffman 코드 생성
    char code[MAX_CODE_LENGTH];
    generateHuffmanCodes(root, code, 0, codes);

    // 결과 출력
    printf("Huffman Code:\n");
    for (int i = 0; i < CHAR_COUNT; i++) {
        printf("%c = %s\n", "ATGC"[i], codes[(int)"ATGC"[i]]);
    }

    // 압축 해제
    decodeHuffmanCode(root, encodedStr);

    return 0;
}

