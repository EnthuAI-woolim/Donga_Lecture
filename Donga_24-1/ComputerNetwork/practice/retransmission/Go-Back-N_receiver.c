#include "../Common.h"

char *SERVERIP = (char *)"127.0.0.1";
#define SERVERPORT 9000
#define BUFSIZE    512
#define PACKET_COUNT 6

// 데이터 통신에 사용할 변수
SOCKET sock;
char buf[BUFSIZE + 1];
int len;
int retval;

int retrans_msg = 0;
int deliver = 0;
int ack_number = 0;
char* packet_msg[PACKET_COUNT] = {"packet 0", "packet 1", "packet 2", "packet 3", "packet 4", "packet 5"};
char* ack_msg[PACKET_COUNT] = {"ACK 0", "ACK 1", "ACK 2", "ACK 3", "ACK 4", "ACK 5"};

// 데이터를 수신하는 함수
int recvData() {
	// 데이터 받기
	retval = recv(sock, buf, BUFSIZE, 0);
	if (retval == SOCKET_ERROR) err_display("recv()");
	else if (retval == 0) return 1;

	return 0;
}

// 데이터를 전송하는 함수
int sendData() {
	strcpy(buf, ack_msg[ack_number++]);
	
	// 데이터 보내기
	retval = send(sock, buf, strlen(buf), 0);
	if (retval == SOCKET_ERROR) {
		err_display("send()");
	}
	if (retrans_msg == 0) printf("\"%s\" is transmitted\n\n", buf);
	else printf("\"%s\" is retransmitted\n\n", buf);

	// 버퍼 초기화
	memset(buf, 0, sizeof(buf));
	sleep(5);
	return 0;
}

int main(int argc, char *argv[])
{
	// 명령행 인수가 있으면 IP 주소로 사용
	if (argc > 1) SERVERIP = argv[1];

	// 소켓 생성
	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == INVALID_SOCKET) err_quit("socket()");

	// connect()
	struct sockaddr_in serveraddr;
	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	inet_pton(AF_INET, SERVERIP, &serveraddr.sin_addr);
	serveraddr.sin_port = htons(SERVERPORT);
	retval = connect(sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
	if (retval == SOCKET_ERROR) err_quit("connect()");

	// 서버와 데이터 통신
	while (1) {
		// 보낼 ACK가 없으면 return
		if (ack_msg[ack_number] == NULL) return 0;

		recvData();

		// 받은 패킷 처리
		if (strcmp(buf, packet_msg[ack_number]) == 0) {
			if (deliver == 0) printf("\"%s\" is received. ", buf);
			else {
				printf("\"%s\" is received and delivered. ", buf);
				retrans_msg = 0;
			}

			// ACK 전송
			sendData();
			
		} else if (strcmp(buf, packet_msg[ack_number]) != 0) {
			printf("\"%s\" is received and dropped. ", buf);
			--ack_number; deliver = 1; retrans_msg = 1;

			// ACK 재전송
			sendData();
			
		} else {
			printf("* Unknown packet received.\n");
		}
		
	}

	// 소켓 닫기
	close(sock);
	return 0;
}