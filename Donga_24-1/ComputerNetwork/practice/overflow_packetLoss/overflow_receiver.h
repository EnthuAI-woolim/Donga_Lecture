#include <ctype.h>
#include <string.h>

char *SERVERIP = (char *)"127.0.0.1";
#define SERVERPORT 9000
#define BUFSIZE    50
#define PACKET_COUNT 5
#define PAYLOAD_SIZE 2

// 데이터 통신에 사용할 변수
SOCKET sock;
int len;
int retval;
int expect_num = 0;
int ack = 0;
int bufferSize = 0;


typedef struct Header {
    unsigned short checksum;   
    int packet_num;        
} Header;

typedef struct Packet {
	struct Header header;
	char payload[PAYLOAD_SIZE];
} Packet;

typedef struct Node {
	Packet packet;
	struct Node* next;
} Node;

typedef struct Queue {
    Node* outputRead;
    Node* front;     
    Node* rear;      
} Queue;



// 큐 초기화 함수
void initQueue(Queue* q) {
    q->outputRead = NULL;
    q->front = NULL;
    q->rear = NULL;
}

// 메세지를 바이너리 형태로 변환하는 함수
unsigned short generate_binary_and_checksum(Packet* packet) {
    int len = strlen(packet->payload);

    char *ptr = packet->payload;
    unsigned short sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += ptr[i];
    }

    return sum;
}

// 모든 비트가 1인지 확인하는 함수
int check_all_ones(unsigned short value) {
    return (value == 0xFFFF);
}

// 큐에 프로세스 추가 함수
void enqueue(Queue* q, Node* newNode) {
    newNode->next = NULL;

    if (q->rear == NULL) {
        q->outputRead = newNode;
        q->front = newNode;
        q->rear = newNode;
    } else {
        q->rear->next = newNode; // 큐에서 마지막 프로세스의 다음 프로세스로 설정
        q->rear = newNode;       // 큐의 마지막 프로세스로 설정
    }
}

// 데이터를 수신하는 함수
void* recvData(void* args) {
    SOCKET sock = ((SOCKET*)args)[0];
    Queue* recvQ = ((Queue**)args)[1];
    Packet buf;

    while (1) {
        // 데이터 받기
        retval = recv(sock, &buf, sizeof(Packet), 0);
        if (retval == SOCKET_ERROR) err_display("recv()");
        else if (retval == 0) return NULL;

        if (sizeof(buf) + bufferSize > BUFSIZE) continue;

        unsigned short binary_data = generate_binary_and_checksum(&buf);
        unsigned short total_sum = buf.header.checksum + binary_data;
        if (check_all_ones(total_sum)) {

            // 받은 데이터 저장하기
            Node* node = (Node*)malloc(sizeof(Node));
            node->packet = buf;
            
            bufferSize += sizeof(buf);

            enqueue(recvQ, node);

            memset(&buf, 0, sizeof(Packet));
                    
        } else {
            printf("checksum 오류입니다.\n");
            break;
        }

    }
	
	return NULL;
}

// 데이터를 전송하는 함수
void* sendData(void* args) {
    SOCKET sock = ((SOCKET*)args)[0];
    Queue* recvQ = ((Queue**)args)[1];
    Packet buf;
    Node* currentNode;
    Packet currentPacket;
    Node* prevNode;

    currentNode = recvQ->front;
    
    while (currentNode != NULL) {

        currentPacket = currentNode->packet;

        if (expect_num == currentPacket.header.packet_num) {
            ack += sizeof(buf);
            expect_num++;

            prevNode = currentNode;
            recvQ->front = currentNode;
            currentNode = currentNode->next;
        } else {
            currentNode = currentNode->next;
            recvQ->front->next = currentNode;
        }

        bufferSize -= sizeof(currentPacket);
        buf.header.packet_num = ack;
        // 데이터 보내기
        retval = send(sock, &buf, BUFSIZE, 0);
        if (retval == SOCKET_ERROR) err_display("send()");

        printf("packet %d is received and there is no error. (%s) (ACK = %d) is transmitted", currentPacket.header.packet_num, currentPacket.payload, ack);

        // 버퍼 초기화  
        memset(&buf, 0, sizeof(Packet));

        sleep(0.1);
        
    }

	return NULL;
}