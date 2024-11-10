#include <stdio.h>
#include <stdlib.h>
#include <string.h>



struct Node { 
    char name;
    int frequency;
    struct Node *left;
    struct Node *right;
    char binary_code[100];
};

typedef struct {
    struct Node *heap[100];
    int size;
} PriorityQueue;

// 최소힙을 위한 비교
int compareNodes(struct Node *a, struct Node *b) { 
    return a->frequency - b->frequency;
}

void swap(struct Node **a, struct Node **b) {
    struct Node *temp = *a;
    *a = *b;
    *b = temp;
}

void init(PriorityQueue *pq) {
    pq->size = 0;
}

void push(PriorityQueue *pq, struct Node *value) {
    if (pq->size >= 100) {
        printf("Queue is full\n");
        return;
    }
    
    int i = pq->size++;
    pq->heap[i] = value;
    // 비교 후, 정렬됨
    while (i > 0 && compareNodes(pq->heap[(i-1)/2], pq->heap[i]) > 0) {
        swap(&pq->heap[(i-1)/2], &pq->heap[i]);
        i = (i-1)/2;
    }
}

struct Node *pop(PriorityQueue *pq) {
    if (pq->size <= 0) {
        printf("Queue is empty\n");
        return NULL;
    }
    
    struct Node *root = pq->heap[0];
    pq->heap[0] = pq->heap[--pq->size];
    
    int i = 0;
    while (1) {
        int smallest = i;
        int left = 2*i + 1;
        int right = 2*i + 2;
        
        if (left < pq->size && compareNodes(pq->heap[left], pq->heap[smallest]) < 0)
            smallest = left;
        if (right < pq->size && compareNodes(pq->heap[right], pq->heap[smallest]) < 0)
            smallest = right;
        
        if (smallest == i)
            break;
        
        swap(&pq->heap[i], &pq->heap[smallest]);
        i = smallest;
    }
    
    return root;
}

/* binary tree function */
void assign_binary_code(struct Node* root, char* code, int depth){ 
    if (root == NULL) return;
    // leaf 일때 바이너리 코드 할당
    if (root->left == NULL && root->right ==NULL) {
        strncpy(root->binary_code, code, depth);
        root->binary_code[depth] = '\0';
        return;
    }
    // left 자식으로 갈때 '0' 추가
    code[depth] = '0';
    assign_binary_code(root->left, code, depth+ 1);

    // right 자식으로 갈때 '1' 추가
    code[depth] = '1';
    assign_binary_code(root->right, code, depth+1);
}

// Decode func
char decode_char(struct Node* root, const char** code_ptr) {
    struct Node* current = root;
    while (current->left != NULL || current->right != NULL) {
        // 0이면 left, 1이면 right
        if (**code_ptr == '0') {
            current = current->left;
        } else if (**code_ptr == '1') {
            current = current->right;
        } else {
            fprintf(stderr, "decode error\n");
            exit(1);
        }
        (*code_ptr)++;
    }
    return current->name;
}

int main() { 
/* 1) Huffman_input.txt 파일을 읽어서 문자열을 입력받는다. */
/* 2) 읽은 파일을 List에 넣는다. */
     FILE *fp = fopen("Huffman_input.txt", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    char input_string[1000];
    if (fgets(input_string, sizeof(input_string), fp) == NULL) {
        perror("Error reading file");
        fclose(fp);
        return 1;
    }
    fclose(fp);

/* 3) input_string 리스트에서 A,T,G,C 를 세알린다.*/ 
    int A_count = 0, T_count = 0, G_count = 0, C_count = 0;
    for (int i = 0; input_string[i] != '\0'; i++) {
        switch(input_string[i]) {
            case 'A': A_count++; break;
            case 'T': T_count++; break;
            case 'G': G_count++; break;
            case 'C': C_count++; break;
        }
    }
/* 4) A,T,G,C Node 생성 및 초기화 */
    struct Node *A_node = (struct Node *)malloc(sizeof(struct Node));
    struct Node *T_node = (struct Node *)malloc(sizeof(struct Node));
    struct Node *G_node = (struct Node *)malloc(sizeof(struct Node));
    struct Node *C_node = (struct Node *)malloc(sizeof(struct Node));

    if (!A_node || !T_node || !G_node || !C_node) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    A_node->name = 'A'; A_node->left = A_node->right = NULL;
    T_node->name = 'T'; T_node->left = T_node->right = NULL;
    G_node->name = 'G'; G_node->left = G_node->right = NULL;
    C_node->name = 'C'; C_node->left = C_node->right = NULL;


    A_node->frequency = A_count;
    T_node->frequency = T_count;
    G_node->frequency = G_count;
    C_node->frequency = C_count;

    printf("A_count: %d\n", A_node->frequency);
    printf("T_count: %d\n", T_node->frequency);
    printf("G_count: %d\n", G_node->frequency);
    printf("C_count: %d\n", C_node->frequency);

    strcpy(A_node->binary_code, "");
    strcpy(T_node->binary_code, "");
    strcpy(G_node->binary_code, "");
    strcpy(C_node->binary_code, "");


/* 5) 빈도수를 우선순위로 하는 Priority_Q 에 노드를 삽입한다. */
    PriorityQueue priority_Q;
    init(&priority_Q);
    push(&priority_Q, A_node);
    push(&priority_Q, T_node);
    push(&priority_Q, G_node);
    push(&priority_Q, C_node);

    while (priority_Q.size > 1) { 
        struct Node *first_node = pop(&priority_Q);
        struct Node *second_node = pop(&priority_Q);

        struct Node *new_node = malloc(sizeof(struct Node));
        if (!new_node) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
        new_node->name = '\0'; // 이름은 없음
        new_node->frequency = first_node->frequency + second_node->frequency;
        new_node->left = first_node;
        new_node->right = second_node;
        strcpy(new_node->binary_code, ""); // 이진코드는 없음
        push(&priority_Q, new_node);
    }
/* 6) 큐에 하나의 노드만 남으면, (== 2개 이하의 노드) 큐를 return 한다. */
    // 루트 노드 출력
    struct Node *root_node = pop(&priority_Q);
    printf("frequency: %d\n", root_node->frequency);
    char b_code[100];
    char *object_string = "10110010001110101010100";
    
    assign_binary_code(root_node, b_code, 0);
    
    printf("A: %s\n", A_node->binary_code);
    printf("T: %s\n", T_node->binary_code);
    printf("G: %s\n", G_node->binary_code);
    printf("C: %s\n", C_node->binary_code);
/*
6) 큐에서 노드 2개를 빼고 부모를 만드는 과정을 거치고, 합친 두개를 각각 자식으로 가르킨다. while 반복을 통해 큐에 하나의 노드만 남으면, (== 2개 이하의 노드) 큐를 return 한다. 

7) 루트에서 시작해서, 리프 노드까지 내려가면서 왼쪽으로 내려가면 0, 오른쪽으로 내려가면 1을 붙인다. 이는 Node마다 binary_code[100]을 붙여 두었기 때문에 가능하다

8) 그리고 A,T,G,C 노드에 이진코드를 붙인다. */

    printf("Encoded string: %s\n", object_string);
    printf("Decoded string: ");
    const char *current_code = object_string;
    
/* 9) decoding을 위해서 기존 만들었던 트리(=루트 노드)와 목표 문자열을 받아서, 0이면 왼쪽, 1이면 오른쪽으로 보낸다. 리프 노드에 도착하면, 해당 노드의 name을 반환한다.*/
    while (*current_code != '\0') {
        printf("%c", decode_char(root_node, &current_code));
    }
    printf("\n");
    
    // 메모리 해제
    free(A_node);
    free(T_node);
    free(G_node);
    free(C_node);

    return 0;
}
