#include <ctype.h>
#include <string.h>

char *SERVERIP = (char *)"127.0.0.1";
#define SERVERPORT 9000
#define BUFSIZE    512
#define PACKET_COUNT 5

typedef struct Packet {
	int id;
	char message[BUFSIZE];
	int ack_num;
	unsigned short checksum;
} Packet;


// 데이터 통신에 사용할 변수
SOCKET sock;
// Packet recvBuf;
Packet recvBuf;
char sendBuf[BUFSIZE + 1];
int len;
int retval;
int expect_pkt = 0;
int retransmission = 0;

int retrans_packet = 0;



int retrans_msg = 0;
int deliver = 0;

// 수신한 모든 패킷을 저장하는 배열
char* received_packets[PACKET_COUNT];


void setPacket(Packet packet[]) {
	for (int i = 0; i < PACKET_COUNT; ++i) {
		packet[i].id = i;
		packet[i].message;
		packet[i].ack_num = 0;
	}
}

// 메세지를 바이너리 형태로 변환하는 함수
unsigned short generate_binary_and_checksum(Packet *packet) {
    int len = strlen(packet->message);

    char *ptr = packet->message;
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

// 데이터를 수신하는 함수
int recvData(Packet packet[]) {
	// 데이터 받기
	retval = recv(sock, &recvBuf, sizeof(Packet), 0);
	if (retval == SOCKET_ERROR) err_display("recv()");
	else if (retval == 0) return 1;

	// 받은 데이터 저장하기
	int packet_id = recvBuf.id;
	strcpy(packet[packet_id].message, recvBuf.message);
	packet[packet_id].checksum = recvBuf.checksum;
	
	size_t byte = 0;
	// 영어와 한글 모두 16비트로 표현하여 바이트 수 계산
	for (size_t i = 0; recvBuf.message[i] != '\0'; ++i) byte += 2;

	if (packet_id == 0) packet[packet_id].ack_num = byte;
	else packet[packet_id].ack_num = packet[packet_id - 1].ack_num + byte;

	printf("packet %d is received and there is no error. (%s) ", packet_id, recvBuf.message);
	
	return 0;
}

// 데이터를 전송하는 함수
int sendData(Packet* packet) {

	sprintf(sendBuf, "%d", packet[expect_pkt++].ack_num);
	
	// 데이터 보내기
	retval = send(sock, sendBuf, BUFSIZE, 0);
	if (retval == SOCKET_ERROR) err_display("send()");
	
	printf("(ACK = %s) is transmitted.\n\n", sendBuf);

	// 버퍼 초기화
	memset(sendBuf, 0, sizeof(sendBuf));
	sleep(5);
	return 0;
}