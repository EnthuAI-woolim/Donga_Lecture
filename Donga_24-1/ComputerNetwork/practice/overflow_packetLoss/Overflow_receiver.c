#include "./Common.h"
#include "./overflow_receiver.h"

#include <pthread.h>


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

	
	// 저장할 packet 정의
	// Packet packet[PACKET_COUNT];
	// setPacket(packet);

	Queue* q = (Queue*)malloc(sizeof(Queue));
	initQueue(q);

	// 스레드 생성 및 실행
	pthread_t send_tid, recv_tid;
	Queue* recvQ = (Queue*)malloc(sizeof(Queue));
	void* args[] = { &sock, recvQ };
	
	initQueue(recvQ);

	if (pthread_create(&recv_tid, NULL, recvData, args)) {
        printf("thread create error\n");
        exit(1);
}
	if (pthread_create(&send_tid, NULL, sendData, args)) {
        printf("thread create error\n");
        exit(1);
	}
	
	
	free(recvQ);
	pthread_join(recv_tid, NULL);
	pthread_join(send_tid, NULL);

	
	// 소켓 닫기
	close(sock);
	return 0;
}