#include <ctype.h>
#include <string.h>

#define SERVERPORT 9000
#define BUFSIZE    512
#define PACKET_COUNT 5
#define TIMEOUT_INTERVER 15

typedef struct Packet {
	int id;
	char message[BUFSIZE];
	int sequence_num;
	unsigned short checksum;
} Packet;


// 데이터 통신에 사용할 변수
SOCKET client_sock;
struct sockaddr_in clientaddr;
socklen_t addrlen;
Packet sendBuf;
char recvBuf[BUFSIZE + 1];
int retval;
int retransmission = 0;
int timeout_count = 0;
int pkt1_error = 0;
int start_index = 0;
int prev_ack = 0;
int cur_ack = 0;
int dup_count = 0;
int recev_ack[PACKET_COUNT];
int count = 0;

size_t start_sequence = 0;

// 메세지를 바이너리 형태로 생성하고 체크섬을 계산하여 구조체에 저장하는 함수
void generate_binary_and_checksum(Packet *packet) {
    int len = strlen(packet->message);

    char *ptr = packet->message;
    unsigned short sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += ptr[i];
    }
    packet->checksum = ~sum;
}

void setPacket(Packet packet[]) {
	int start_sequence = 0;

	strcpy(packet[0].message, "I am a boy.");
	strcpy(packet[1].message, "You are a girl.");
	strcpy(packet[2].message, "There are many animals in the zoo.");
	strcpy(packet[3].message, "철수와 영희는 서로 좋아합니다!");
	strcpy(packet[4].message, "나는 점심을 맛있게 먹었습니다.");

	for (int i = 0; i < PACKET_COUNT; ++i) {
		packet[i].id = i;
		packet[i].sequence_num = start_sequence;

		// 영어와 한글 모두 16비트로 표현하여 바이트 수 계산
		for (size_t j = 0; packet[i].message[j] != '\0'; ++j) start_sequence += 2;
		
		generate_binary_and_checksum(&packet[i]);
	}
}

int findIndex(Packet* packet, int prev_ack) {
	for (int i = 0; i < PACKET_COUNT; ++i) 
		if (packet[i].sequence_num == prev_ack) return packet[i].id;
}

// 데이터를 수신하는 함수
int recvData(Packet *packet) {

	// "packet 1"를 처음 보낼 경우 recevier에 전송안함
	if (packet->id == 1 && pkt1_error == 0) {
		pkt1_error = 1;
		return 0;
	}

	// 데이터 받기
	retval = recv(client_sock, recvBuf, BUFSIZE, 0);
	if (retval == SOCKET_ERROR) err_display("recv()");
	else if (retval == 0) return 1;
	
	// 문자열에서 추출한 정수를 배열에 저장
	recev_ack[count++] = atoi(recvBuf);

	return 0;
}

// 데이터를 전송하는 함수
int sendData(Packet *packet) {
	
	memcpy(&sendBuf, packet, sizeof(Packet)); // 구조체를 바이트 배열로 복사

    if (retransmission == 1) printf("packet %d is retransmitted. (%s)\n\n", packet->id, packet->message);
	else printf("packet %d is transmitted. (%s)\n\n", packet->id, packet->message);

	timeout_count++;
	// "packet 1"를 처음 보낼 경우 recevier에 전송안함
	if (packet->id == 1 && pkt1_error == 0) {
		sleep(5);
		return 0;
	}

	// 데이터 보내기
	retval = send(client_sock, &sendBuf, sizeof(Packet), 0);
	if (retval == SOCKET_ERROR) err_display("send()");
	
	memset(&sendBuf, 0, sizeof(Packet));
	
	sleep(5);
	
	return 0;
}