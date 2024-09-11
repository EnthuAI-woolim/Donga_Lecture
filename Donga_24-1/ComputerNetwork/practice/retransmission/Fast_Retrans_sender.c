#include "../Common.h"
#include "./FR_sender.h"


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
		
		// 패킷 생성 및 설정
		Packet packet[PACKET_COUNT];
		setPacket(packet);

		// 클라이언트와 데이터 통신
		while (1) {
			
			// Packet 전송
			for (int i = start_index; i < PACKET_COUNT; ++i) { 
				sendData(&packet[i]);
				if (retransmission == 1) break; 

				recvData(&packet[i]);
			}

			if (retransmission == 1) break; 
			
			// 받은 ACK 처리
			for (int i = 0; i < count; ++i) {
				prev_ack = cur_ack;
				cur_ack = recev_ack[i];
				
				printf("(ACK = %d) is received.\n\n", cur_ack);

				// duplicate ACK 횟수 설정
				if (cur_ack == prev_ack) dup_count++;

				if (dup_count == 3) {
					start_index = findIndex(packet, prev_ack);	// 시작인덱스 재설정
					retransmission = 1;
					continue;
				}
			}
		}
		
		// 소켓 닫기
		close(client_sock);
		break;
	}

	// 소켓 닫기
	close(listen_sock);
	return 0;
}