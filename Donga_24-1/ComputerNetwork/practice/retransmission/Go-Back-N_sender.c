#include "../Common.h"
#include <ctype.h>
#include <string.h>

#define SERVERPORT 9000
#define BUFSIZE    512
#define PACKET_COUNT 6
#define MAX_RECV 100
#define TIMEOUT_INTERVER 5

// 데이터 통신에 사용할 변수
SOCKET client_sock;
struct sockaddr_in clientaddr;
socklen_t addrlen;
char buf[BUFSIZE + 1];
int retval;

int timeout_count = 0;
int pkt2_error = 0;
int packet_number = 0;
int ack_number = 0;
char* packet_msg[PACKET_COUNT] = {"packet 0", "packet 1", "packet 2", "packet 3", "packet 4", "packet 5"};
char* ack_msg[PACKET_COUNT] = {"ACK 0", "ACK 1", "ACK 2 ", "ACK 3", "ACK 4", "ACK 5"};
char recv_ack[MAX_RECV][10];

// 인덱스 찾는 함수
int findLen() {
	int count = 0;
  for (int i = 0; i < MAX_RECV; i++) 
    if (recv_ack[i][0] != '\0') count++;
    else break; // 널 종단 문자를 만나면 루프 종료
    
  return count;
}

// 데이터를 수신하는 함수
int recvData() {

  memset(buf, 0, sizeof(buf));

	// 데이터 받기
	retval = recv(client_sock, buf, BUFSIZE, 0);
	if (retval == SOCKET_ERROR) err_display("recv()");
	else if (retval == 0) return 1;

  int len = strlen(buf);
  int index = 0;
  // recevier에서 받은 ACK를 recv_ack에 저장
  for (int i = 0; i < len; i += 5) {
      strncpy(recv_ack[index], buf + i, 5); // 현재 위치에서 4개의 문자를 복사
      recv_ack[index++][5] = '\0'; // 널 문자 추가
  }
	return 0;
}

// 데이터를 전송하는 함수
int sendData() {
	strcpy(buf, packet_msg[packet_number++]);

	printf("\"%s\" is transmitted\n\n", buf);
	
	timeout_count++;
	// "packet 2"를 처음 보낼 경우 recevier에 전송안함
	if (strcmp(buf, packet_msg[2]) == 0 && pkt2_error == 0) {
		pkt2_error++;
		sleep(5);
		return 0;
	}

	// 데이터 보내기
	retval = send(client_sock, buf, strlen(buf), 0);
	if (retval == SOCKET_ERROR) {
		err_display("send()");
	}
	// 버퍼 초기화
	memset(buf, 0, sizeof(buf));

	sleep(5);
	return 0;
}

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
		
		// 클라이언트와 데이터 통신
		while (1) {
			
			// 패킷 4개 보내기
			for (int i = 0; i < 4; i++) {
				sendData();
				
			}

			// 보낼 패킷이 없을 경우 return
			if (packet_msg[packet_number] == NULL) return 0;

      recvData();

			int len = findLen();
			// 받은 패킷에 대해 처리
      for (int i = 0; i < len; i++) {
        if (strcmp(recv_ack[i], ack_msg[ack_number]) == 0) {
					
          printf("\"%s\" is received. ", recv_ack[i]);

          sendData();

          ack_number++; timeout_count--;
        } else if (strcmp(recv_ack[i], ack_msg[ack_number]) != 0) {
          printf("\"%s\" is received and ignored.\n\n", recv_ack[i]);
					timeout_count++;

					// timeout
					if (timeout_count == TIMEOUT_INTERVER) {
						printf("\"%s\" is timeout.\n\n", packet_msg[ack_number]);
						packet_number = ack_number; // 재전송해야되는 패킷의 인덱스 재정의

						// 받은 ACK가 저장되어 있는 변수 리셋, TimeoutInterver 리셋
						for (int i = 0; i < len; i++) memset(recv_ack[i], 0, sizeof(recv_ack[i]));
						timeout_count = 0;
						break;
					}
				}
      }
		}

		// 소켓 닫기
		close(client_sock);
	}

	// 소켓 닫기
	close(listen_sock);
	return 0;
}