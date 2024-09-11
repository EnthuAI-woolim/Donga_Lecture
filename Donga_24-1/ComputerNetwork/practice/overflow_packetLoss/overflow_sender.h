#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>

#define SERVERPORT 9000
#define BUFSIZE    512
#define PAYLOAD_SIZE 2
#define WINDOW_SIZE 4
#define TIMEOUT_INTERVER 0.5

typedef struct Header {
    unsigned short checksum;   
    int packet_number;        
} Header;

typedef struct Packet {
	struct Header header;
	char payload[PAYLOAD_SIZE];
} Packet;

typedef struct Node {
	Packet packet;
    int timeout;
    int ack;
	struct Node* next;
} Node;

typedef struct Queue {
    Node* front;     
    Node* rear;      
} Queue;




// 데이터 통신에 사용할 변수
SOCKET client_sock;
struct sockaddr_in clientaddr;
socklen_t addrlen;
int retval;

FILE* file;
Node* timeoutNode;
Packet buf;
int windowSize;


// 큐 초기화 함수
void initQueue(Queue* q) {
    q->front = NULL;
    q->rear = NULL;
}

// 메세지를 바이너리 형태로 생성하고 체크섬을 계산하여 구조체에 저장하는 함수
void generate_binary_and_checksum(Packet *packet) {
    int len = strlen(packet->payload);

    char *ptr = packet->payload;
    unsigned short sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += ptr[i];
    }
    packet->header.checksum = ~sum;
}

// 큐에 프로세스 추가 함수
void enqueue(Queue* sendQ, Node* newNode) {
    newNode->next = NULL;

    if (sendQ->rear == NULL) {
        sendQ->front = newNode;
        sendQ->rear = newNode;
    } else {
        sendQ->rear->next = newNode; // 큐에서 마지막 프로세스의 다음 프로세스로 설정
        sendQ->rear = newNode;       // 큐의 마지막 프로세스로 설정
    }
}

void setPacketBuffer_enQueue(Queue* sendQ) {
	char* filename = "text.txt";
	int num = 0;
    int ack_num = 0;
    int index = 0;
    char ch;
    wchar_t payload[512];

    setlocale(LC_ALL, "");
    
    // 파일 열기
    if((file = fopen(filename, "r")) == NULL) {
        printf("ERROR - cannot open file\n");
        return;
    }

    // UTF-16LE 형식으로부터 두 글자 읽어오기
    wchar_t ch1, ch2;
    if (fread(&ch1, sizeof(wchar_t), 1, file) != 1) {
        printf("ERROR - cannot read from file\n");
        fclose(file);
        return;
    }
    if (fread(&ch2, sizeof(wchar_t), 1, file) != 1) {
        printf("ERROR - cannot read from file\n");
        fclose(file);
        return;
    }
    
    
    

    //     // 큐에 추가
    //     enqueue(sendQ, node);


    // // 파일 닫기
    // fclose(file);
}

void* sendData(void* args) {
    SOCKET client_sock = ((SOCKET*)args)[0];
    Queue* sendQ = ((Queue**)args)[1];
    
    Node* currentNode;
    Packet buf;
    
    while (1) {
        timeoutNode = sendQ->front;
        currentNode = timeoutNode;

        while (currentNode != NULL && windowSize != 0) {
            if (timeoutNode->timeout == 15) {
                timeoutNode->timeout = 0;
                break;
            }

            buf = currentNode->packet;

            // 데이터 보내기
            retval = send(client_sock, &buf, sizeof(Packet), 0);
            if (retval == SOCKET_ERROR) err_display("send()");

            printf("packet %d is transmitted. (%s)\n\n", buf.header.packet_number, buf.payload);

            memset(&buf, 0, sizeof(Packet));

            windowSize--;
            currentNode = currentNode->next;
            timeoutNode->timeout += 0.05;
            sleep(0.05);
        }

        if (currentNode == NULL) break;
    }

    pthread_exit(NULL);
    return NULL;
}

void* recvData(void* args) {
    SOCKET client_sock = ((SOCKET*)args)[0];
    Queue* sendQ = ((Queue**)args)[1];
	Packet buf;

    while(sendQ->front != NULL) {
        timeoutNode = sendQ->front;

        // 데이터 받기
        retval = recv(client_sock, &buf, sizeof(Packet), 0);
        if (retval == SOCKET_ERROR) err_display("recv()");
        else if (retval == 0) return NULL;

        int recv_ack = buf.header.packet_number;

        if (timeoutNode->ack != recv_ack) {
            printf("(ACK = %d) is received and ignored\n\n", recv_ack);
            continue;
        }

        printf("(ACK = %d) is received and ", recv_ack);
        timeoutNode->next->timeout = timeoutNode->timeout;
        sendQ->front = timeoutNode->next;
        
        windowSize = WINDOW_SIZE;

        memset(&buf, 0, sizeof(Packet));

    }

	pthread_exit(NULL);
}