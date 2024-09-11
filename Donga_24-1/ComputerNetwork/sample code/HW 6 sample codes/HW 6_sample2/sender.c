#include "Common.h"

#define SERVERPORT 9000
#define PACKETSIZE    10
#define MSGSIZE 3
#define T 0.05
#define WNDSIZE 4
#define TIMEOUT 0.5

char *SERVERIP = (char *)"127.0.0.1";
SOCKET sock;
FILE* fp;
char txt[513]; // txt 파일을 읽어올 버퍼

char message[7]; // 메세지를 읽는 버퍼
char packet[11]; // 패킷을 생성하는 버퍼

int S = 1; // 스레드를 제어하는 변수
Information info[4];
int indicator = 0;
int timeout_packet = -1;
int duplicate = 0;
int oldACK = 0;
int ACK = 0;

/*
** 함수부 **
checksum 계산
스레드 함수
Sender - 문자열 중 일부를 읽어 패킷을 만들고 전송
Receiver - ACK을 읽어 정보를 업데이트
*/
// checksum 계산을 위한 함수 정의
void add_binary(int* checksum, int add[9]);
unsigned char* calculate_checksum(char packet[7]);

// 스레드 함수
void* send_thread(void* arg);
void* recv_thread(void* arg);

int main(int argc, char *argv[])
{
	// txt 파일 읽기
	fp = fopen("text.txt", "r");
	fgets(txt, 512, fp); 
	txt[strlen(txt)-1] = '\0';

	info[0].end = -1;
	info[1].end = -1;
	info[2].end = -1;
	info[3].end = -1;
	pthread_t send_id, recv_id;

	int retval;

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

	// 데이터 통신에 사용할 변수
	int len;

	while(1) {
		
		pthread_create(&send_id, NULL, send_thread, NULL);
		pthread_create(&recv_id, NULL, recv_thread, NULL);

		pthread_join(send_id, NULL);
		pthread_join(recv_id, NULL);
	}
	fclose(fp);

	// 소켓 닫기
	close(sock);
	return 0;
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

unsigned char* calculate_checksum(char packet[7]) {
	int mask;
	static unsigned char check[2];

	int checksum[9] = {0, };
	for (int i = 0; i < strlen(packet); i++) {
		int val[9] = {0,};
		for (int j = 7; j >= 0; j--) {
			mask = 1 << j;
			val[8-j] = packet[i] & mask ? 1 : 0;
		};
		add_binary(checksum, val);
		
		if (checksum[0] == 1) {
			int v[9] = {0};
			v[8] = 1;
			add_binary(checksum, v);
			checksum[0] = 0;
		}
	}
	for (int j = 1; j < 9; j++) {
		checksum[j] = checksum[j] == 1 ? 0 : 1;
	}

	// 2진수 -> 16진수로 변환
	check[0] = 0b00000000;
	check[1] = '\0';
	
	int n = 7;
	int c = 0;
	for(int i = 0; i < 8; i++) {
		c += checksum[i+1] * (int)(pow(2, n));
		n--;
	}

	check[0] = check[0] | c;

	return check;
}

void* send_thread(void* arg) {

	while(1) {
		if (S != 1) break;
		if (S != 0) {
			int start;

			// get message from txt file
			message[0] = '\0';
			int loc = 0;
			for (int word = 0; word < 2; word++) {
				start = info[indicator%4].end+1;
				
				if (txt[start] & 0x80) {
					message[loc++] = txt[start++];
					message[loc++] = txt[start++];
					message[loc++] = txt[start++];
				} else {
					message[loc++] = txt[start++];
				}

				info[indicator%4].end = start-1;
			}
			message[loc]='\0';
			if (!strcmp(message, "")) {
				printf("\b\b\b\b\b.   \n");
				S = 0;
				break;
			}
			// calculate checksum
			unsigned char checksum1[2];
			unsigned char* c;
			c = calculate_checksum(message);
			checksum1[0] = c[0];
			checksum1[1] = '\0';

			if (message[3] == ' ') {
				message[3] = '0';
			}

			// make packet for send
			packet[0] = '\0';
			sprintf(packet, "%2d%2x%6s", info[indicator%4].indicator, checksum1[0], message);
			packet[strlen(packet)] = '\0';

			if (message[3] == '0') message[3] = ' ';

			// send packet
			int retval = send(sock, packet, (int)strlen(packet), 0);
			if (retval == SOCKET_ERROR) {
				err_display("send()");
				break;
			}
			printf("packet %d is transmitted. (%s)\n", info[indicator%4].indicator, message);
			sleep(T);
			info[indicator%4].timeout = clock();

			// calculate now send_base and nextseqnum
			info[indicator%4].nextseqnum += (int)strlen(packet);
			info[indicator%4].indicator++;
			if (info[indicator%4].send_base + WNDSIZE*PACKETSIZE < info[indicator%4].nextseqnum + PACKETSIZE) {
				S = 0;
			}

			// 보낼 텍스트가 없으면 종료
			if (info[indicator%4].end >= (int)strlen(txt)-1)
				break;
			
		}
	}
	pthread_exit(NULL);
}
void* recv_thread(void* arg) {
	while(1) {
		if (S != 0) break;
		if (S != 1) {
			
			// timeout 검사
			clock_t endtime = clock();
			for (int i = 0; i < 4; i++) {
				if ((double)(endtime - info[i].timeout) / CLOCKS_PER_SEC > TIMEOUT) {
					timeout_packet = info[i].indicator;
					printf("packet %d is timeout.\n", timeout_packet);
				}
			}

			// ack 메세지 수신
			char buf[4];
			int retval = recv(sock, buf, 4, MSG_WAITALL);
			if (retval == SOCKET_ERROR) {
				err_display("recv()");
				break;
			}
			else if (retval == 0)
				break;
			
			ACK = atoi(buf);

			// 중복 ack인지 검사
			if (oldACK == ACK) duplicate = 1;
			printf("(ACK = %d) is received and ", ACK);

			oldACK = ACK;
			if (duplicate == 0) {
				// ack이 제대로 수신된 경우 send_base 증가
				info[indicator%4].send_base += PACKETSIZE;
				if (info[indicator%4].send_base + WNDSIZE*PACKETSIZE >= info[indicator%4].nextseqnum + PACKETSIZE) {
					S = 1;
					break;
				}
			} else {
				printf("ignored.\n");
				S = 0;
			}
		}
	}

	pthread_exit(NULL);
}