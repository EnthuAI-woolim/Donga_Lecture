#include <stdio.h>
#include <stdlib.h>
#include <string.h>  //문자열 처리를 위한 헤더 추가
#define ASCII_SIZE 128 //전체 아스키 코드에 대한 크기
#define MAX_SIZE 100 //우선순위 큐의 최대 사이즈

typedef struct HuffmanNode { //허프만 트리의 노드에 대한 구조체
    int frequency; //문자의 빈도 수
    char symbol; //그 문자 그 자체
    struct HuffmanNode* left; //현재 노드의 좌측 노드
    struct HuffmanNode* right; //현재 노드의 우측 노드
} HuffmanNode;

typedef struct { //
    HuffmanNode* heap[MAX_SIZE];
    int heap_size;
} HeapType;

void init(HeapType* h) { //우선순위 큐를 초기화
    h->heap_size = 0;
}

void insert(HeapType* h, HuffmanNode* item) { //우선순위 큐에 허프만 노드를 삽입하는 함수.
    int i = ++(h->heap_size);

    while (i != 1 && item->frequency < h->heap[i / 2]->frequency) { //현재 삽입한 노드가 부모노드보다 빈도수가 더 낮다면
        h->heap[i] = h->heap[i / 2]; //부모노드를 아래쪽으로 내림
        i /= 2; //인덱스 조절
    }
    h->heap[i] = item; //제위치를 찾았으면 그자리에 item을 넣어준다.
}

HuffmanNode* pop(HeapType* h) { //우선순위 큐 pop 함수
    int parent, child;
    HuffmanNode* item = h->heap[1]; //pop해서 나오는 함수
    HuffmanNode* temp = h->heap[(h->heap_size)--]; //우선순위 큐에서 가장 마지막에 있는 노드

    parent = 1;
    child = 2;
    while (child <= h->heap_size) { //자식 노드의 인덱스가 우선순위 큐 사이즈보다 작은 동안 반복
        if(child < h->heap_size && h->heap[child]->frequency > h->heap[child + 1]->frequency)
            child++; //왼쪽, 오른쪽 자식 중 더 작은 자식을 고름. 이 조건문이 참이라면 오른쪽 자식이 더 작다는 뜻
        if(temp->frequency <= h->heap[child]->frequency) break; //temp의 빈도수가 child의 빈도수보다 작다는 뜻은 temp가 올바른 위치까지 내려갔다는 뜻.
        //한레벨 더 내려감
        h->heap[parent] = h->heap[child]; 
        parent = child;
        child *= 2;
    }
    //while문 탈출했다는 뜻은 temp가 올바른 위치까지 왔다는 뜻
    h->heap[parent] = temp;
    return item;
}


HuffmanNode* createHuffmanNode(int frequency, char symbol, HuffmanNode* left, HuffmanNode* right) { //허프만 노드 생성 함수
    HuffmanNode* newNode = (HuffmanNode*)malloc(sizeof(HuffmanNode));
    //주어진 값들로 노드 초기화
    newNode->frequency = frequency;
    newNode->symbol = symbol;
    newNode->left = left;
    newNode->right = right;
    return newNode;
}


HuffmanNode* buildHuffmanTree(HeapType* h) { //허프만 트리 생성 함수
    while (h->heap_size > 1) { //우선순위 큐에 노드가 2개 이상 있는 동안 반복해서 실행(1개 남을 때 까지)
        HuffmanNode* leftTree = pop(h); //첫번째로 빈도수가 작은 노드
        HuffmanNode* rightTree = pop(h); //두번째로 빈도수가 작은 노드

        int frequencySum = leftTree->frequency + rightTree->frequency; //두 빈도수의 합
        HuffmanNode* parent = createHuffmanNode(frequencySum, '\0', leftTree, rightTree); //parent라는 새로운 노드로 만들고, 좌측, 우측 서브트리로 넣는다.

        insert(h, parent); //우선순위 큐에 방금 만든 parent 노드를 삽입한다.
    }
    return pop(h);
}

void generateHuffmanCode(HuffmanNode* root, char* code, int depth) { //허프만 코드를 생성하는 함수
    if(!root) return; //root노드가 NULL이면 바로 종료한다. (만들게 없음)

    if(root->left == NULL && root->right == NULL) { //리프노드이면 심볼과 코드를 출력한다.
        code[depth] = '\0'; //문자열의 종료를 알리는 \0 문자
        printf("Symbol: %c, Code: %s\n", root->symbol, code);
        return;
    }

    code[depth] = '0'; //왼쪽 자식은 0
    generateHuffmanCode(root->left, code, depth + 1); //depth는 1 증가 (문자열 인덱스를 한 칸 옆으로 옮기는 것)

    code[depth] = '1'; //오른쪽 자식은 1
    generateHuffmanCode(root->right, code, depth + 1);
}


void decodeHuffmanCode(HuffmanNode* root, const char* encodedStr) { //디코딩 함수 
    HuffmanNode* current = root;
    printf("Decoded string: ");

    for(int i = 0; i < strlen(encodedStr); i++) { //문자열 끝까지 탐색
        if(encodedStr[i] == '0') { //0을 만나면 왼쪽 트리로 이동
            current = current->left;
        }
        else if(encodedStr[i] == '1') { //1을 만나면 오른쪽 트리로 이동
            current = current->right;
        }
        if(current->left == NULL && current->right == NULL) { //리프노드를 만나면 코드를 출력.(더이상 뻗어나갈 서브트리가 존재하지 않음)
            printf("%c", current->symbol);
            current = root;  //루트로 돌아가서 다음 문자를 찾기 시작
        }
    }
    printf("\n");
}

int main() {
    FILE *file = fopen("Huffman_input.txt", "r");
    if(file == NULL) {
        perror("Error opening file");
        return 1;
    }

    int frequency[ASCII_SIZE] = {0}; //문자별 빈도수를 저장 할 배열
    int ch; //읽어들인 문자를 아스키코드의 10진수 형태로 받아들임
    while ((ch = fgetc(file)) != EOF) {
        if(ch >= 0 && ch < ASCII_SIZE) {
            frequency[ch]++; //해당 문자의 빈도수 증가
        }
    }
    fclose(file);

    HeapType heap; //우선순위 큐 선언
    init(&heap); //초기화

    for(int i = 0; i < ASCII_SIZE; i++) {
        if(frequency[i] > 0) { //해당 문자가 한번이라도 나왔으면 빈도수는 0보다 크다다
            //우선순위큐에 허프만 노드를 생성해서 넣는다. 이 때 아직 모든 노드들은 서브트리를 가지고 있지 않으므로 서브트리들은 모두 NULL로 넣어준다.
            HuffmanNode* newNode = createHuffmanNode(frequency[i], (char)i, NULL, NULL); //i는 해당 문자의 10진수이고, char형으로 바꿔서 넣어준다.
            insert(&heap, newNode);
        }
    }
    HuffmanNode* huffmanTree = buildHuffmanTree(&heap); //허프만 트리를 만든다. 결과는 1개의 트리와 그 서브트리들로 구성된다.

    char code[MAX_SIZE];
    generateHuffmanCode(huffmanTree, code, 0); //해당 문자들에 맞게 코드를 만든다

    //디코딩할 이진수
    const char* encodedStr = "10110010001110101010100";
    decodeHuffmanCode(huffmanTree, encodedStr);

    return 0;
}
