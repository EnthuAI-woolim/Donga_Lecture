#include "./Common.h"
#include "./overflow_sender.h"

#include <pthread.h>


int main(int argc, char *argv[])
{
	// 소켓 생성
	SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, 0);
	if (listen_sock == INVALID_SOCKET) err_quit("socket()");

	// bind()
	struct sockaddr_in serveraddr;
	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
	serveraddr.sin_port = htons(SERVERPORT);
	retval = bind(listen_sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
	if (retval == SOCKET_ERROR) err_quit("bind()");

	// listen()
	retval = listen(listen_sock, SOMAXCONN);
	if (retval == SOCKET_ERROR) err_quit("listen()");

	while (1) {
		// accept()
		addrlen = sizeof(clientaddr);
		client_sock = accept(listen_sock, (struct sockaddr *)&clientaddr, &addrlen);
		if (client_sock == INVALID_SOCKET) {
			err_display("accept()");
			break;
		}
		// 스레드 생성 및 실행
		pthread_t send_tid, recv_tid;
		Queue* sendQ = (Queue*)malloc(sizeof(Queue));
	    void* args[] = { &client_sock, sendQ };
      
		initQueue(sendQ);
		setPacketBuffer_enQueue(sendQ);
		
		if (pthread_create(&send_tid, NULL, sendData, args)) {
			printf("thread create error\n");
			exit(1);
		}
		if (pthread_create(&recv_tid, NULL, recvData, args)) {
			printf("thread create error\n");
			exit(1);
		}
		
    	free(sendQ);
		pthread_join(send_tid, NULL);
		pthread_join(recv_tid, NULL);
   }
		
	
	// 소켓 닫기
	close(client_sock);
	close(listen_sock);
	return 0;
}