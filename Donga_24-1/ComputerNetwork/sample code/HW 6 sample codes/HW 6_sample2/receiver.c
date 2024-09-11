#include "Common.h"

#define SERVERPORT 9000
#define BUFSIZE    50

#define PACKETSIZE 10
#define MSGSIZE 6
#define T 0.1

// 통신을 위한 변수
SOCKET client_sock;
FILE* fp;

char msgbuf[BUFSIZE+1];
int S = 1; // 수신받는 경우 1, 송신하는 경우 0

int ACK;
int oldACK;
int isdropped;

// checksum 계산을 위한 함수 정의
void add_binary(int* checksum, int add[9]);
int calculate_checksum(char packet[MSGSIZE+1], unsigned char* check);

// 스레드 함수
void* send_thread(void* arg); // ack을 전송하는 스레드
void* recv_buffer(void* arg); // 패킷을 읽어서 버퍼에 넣는 스레드
void* recv_thread(void* arg); // 버퍼에서 패킷을 읽어오는 스레드

int main(int argc, char *argv[])
{
	int retval;

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

	// 데이터 통신에 사용할 변수
	struct sockaddr_in clientaddr;
	socklen_t addrlen;

	fp = fopen("output.txt", "w");
	
	while (1) {
		// accept()
		addrlen = sizeof(clientaddr);
		client_sock = accept(listen_sock, (struct sockaddr *)&clientaddr, &addrlen);
		if (client_sock == INVALID_SOCKET) {
			err_display("accept()");
			break;
		}

		// 클라이언트와 데이터 통신
		
		pthread_t send_id, recv_buffer_id, recv_id;
		while (1) {
			pthread_create(&send_id, NULL, send_thread, NULL);
			pthread_create(&recv_buffer_id, NULL, recv_buffer, NULL);
			pthread_create(&recv_id, NULL, recv_thread, NULL);

			pthread_join(send_id, NULL);
			pthread_join(recv_buffer_id, NULL);
			pthread_join(recv_id, NULL);
		}

		// 소켓 닫기
		close(client_sock);
		fclose(fp);
	}

	// 소켓 닫기
	close(listen_sock);
	return 0;
}

void* send_thread(void* arg) {
	while (1) {
		if (S != 1) {
			char buf[4];
			buf[0] = '\0';
			sprintf(buf, "%3d", ACK);
			buf[3] = '\0';

			// ACK 전송
			int retval = send(client_sock, buf, 4, 0);
			if (retval == SOCKET_ERROR) {
				err_display("send()");
				break;
			}

			printf("(ACK = %d) is transmitted.\n", ACK);
			S = 1;
		}
	}

	pthread_exit(NULL);
}
void* recv_buffer(void* arg) {
	while(1) {
		char buf[PACKETSIZE+1];
		buf[0] = '\0';
		int retval = recv(client_sock, buf, PACKETSIZE, 0);
		if (retval == SOCKET_ERROR) {
			err_display("recv()");
			break;
		}
		else if (retval == 0)
			break;

		int packet_len = (int)strlen(buf);
		buf[packet_len] = '\0';

		if (packet_len + strlen(msgbuf) > BUFSIZE) {
			// 패킷이 드랍되는 경우 처리
			isdropped = 1;
			printf("packet is dropped!\n");
		} else {
			strcat(msgbuf, buf);
		}
	}

	pthread_exit(NULL);
}

void* recv_thread(void* arg) {
	while(1) {
		if (S != 0) {
			char packet[PACKETSIZE+1];
			char message[MSGSIZE+1];
			unsigned char checksum[2];
			char ind[3];
			int indicator;
			if ((int)strlen(msgbuf) > 0) {
				strncpy(packet, msgbuf, PACKETSIZE);
				for(int i = 0; i < (int)strlen(msgbuf); i++) {
					if (i + 10 <= (int)strlen(msgbuf))
						msgbuf[i] = msgbuf[i+10];
					else msgbuf[i] = '\0';
				}

				// 패킷을 분리 -> indicator, checksum, message
				for (int i = 0; i < 2; i++) {
					ind[i] = packet[i];
				}
				for (int i = 0; i < 2; i++) {
					checksum[i] = packet[i+2];
				}
				for (int i = 4; i < 11; i++) {
					message[i-4] = packet[i];
					if(message[i-4] == '0') {
						message[i-4] = ' ';
						message[i-3] = '\0';
					}
				}
				indicator = atoi(ind);

				printf("packet %d is received", indicator);
				
				// checksum 확인
				int check = calculate_checksum(message, checksum);
				if (check != 1) {
					// bit error -> drop
					isdropped = 1;
					printf(" and there is some error.");
				} else {
					printf(" and there is no error. (%s) ", message);
					fp = fopen("output.txt", "a");
					if (fp != NULL) {
						int f = fputs(message, fp);
						if (!strcmp(message, ".")) {
							f = fputs("\n", fp);
						}
						fclose(fp);
					}
				}

				if (oldACK != indicator * (int)strlen(packet)) {
					isdropped = 1;
				}

				// ACK을 생성
				if (isdropped == 1) {
					ACK = oldACK;
				} else {
					ACK = indicator * PACKETSIZE + (int)strlen(packet);
				}

				oldACK = ACK;
				S = 0;
				sleep(T);
			}
			
		}
	}

	pthread_exit(NULL);
}

void add_binary(int* checksum, int add[9]) {
	int flag[9] = {0,};
	for (int i = 9; i >= 0; i--) {
		if(checksum[i] + add[i] + flag[i] == 2) {
			checksum[i] = 0;
			flag[i-1] = 1;
		} else if (checksum[i] + add[i] + flag[i] == 3) {
			checksum[i] = 1;
			flag[i-1] = 1;
		} else {
			checksum[i] = checksum[i] + add[i] + flag[i];
		}
	}
}


int calculate_checksum(char packet[MSGSIZE+1], unsigned char* check) {
	int mask;
	int isEqual = 1;
	for (int i = 0; i < (int)strlen(packet); i++) {
		if (!(packet[i] & 0x80)) {
			if (packet[5] == '.' && packet[0] == ' ') {
				packet[0] = '.';
				packet[1] = '\0';
				break;
			}
			int n = 0;
			for (int j = 0; j < 6; j++) {
				if (packet[j] & 0x80) {
					n = j;
					break;
				}
			}
			if (n == 3) {
				for (int j = i+1; j < (int)strlen(packet); j++) {
					packet[j] = packet[j+2];
				}
			} else {
				for (int j = i; j < (int)strlen(packet); j++) {
					packet[j] = packet[j+2];
				}
				packet[3] = ' ';
			}
			packet[4] = '\0';
			break;
		}
	}

	int c[9] = {0, };
	for (int i = 0; i < strlen(packet); i++) {
		int val[9] = {0,};
		for (int j = 7; j >= 0; j--) {
			mask = 1 << j;
			val[8-j] = packet[i] & mask ? 1 : 0;
		};
		add_binary(c, val);
		
		if (c[0] == 1) {
			int v[9] = {0};
			v[8] = 1;
			add_binary(c, v);
			c[0] = 0;
		}
	}

	int checkfrom[9] = {0,};
	int dec = 0;
	for (int i = 0; i < 2; i++) {
		if (check[i] >= 'a')
			dec += (check[i] - 'a' + 10) * (int)(pow(16, 1-i));
		else
			dec += (check[i] - '0') * (int)(pow(16, 1-i));
	}
	
	for (int i = 1; i < 9; i++) {
		if (dec % 2 == 1) checkfrom[9-i] = 1;
		else checkfrom[9-i] = 0;
		if (checkfrom[9-i] + c[9-i] != 1) isEqual = 0;
		dec /= 2;
	}
	return isEqual;
}