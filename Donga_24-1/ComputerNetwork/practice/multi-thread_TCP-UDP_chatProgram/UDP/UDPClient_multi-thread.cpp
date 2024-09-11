#include "../Common.h"

char *SERVERIP = (char *)"127.0.0.1";
#define SERVERPORT 9000
#define BUFSIZE    512

void* sendData(void* arg);
void* recvData(void* arg);

// 데이터 통신에 사용할 변수
SOCKET sock;
struct sockaddr_in peeraddr;
struct sockaddr_in serveraddr;
socklen_t addrlen;
char buf[BUFSIZE + 1];
int retval;
int len;
int result;
int turn = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; // mutex 초기화


int main(int argc, char *argv[])
{
	pthread_t tid1, tid2;

	// 명령행 인수가 있으면 IP 주소로 사용
	if (argc > 1) SERVERIP = argv[1];

	// 소켓 생성
	sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock == INVALID_SOCKET) err_quit("socket()");

	// 소켓 주소 구조체 초기화

	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	inet_pton(AF_INET, SERVERIP, &serveraddr.sin_addr);
	serveraddr.sin_port = htons(SERVERPORT);

	
	// 서버와 데이터 통신
	while (1) {
		if (turn == 0) {
			// 보내는 스레드가 실행되기 전에 mutex 잠금
			pthread_mutex_lock(&mutex);

			int result = pthread_create(&tid1, NULL, sendData, NULL);
			if (result != 0) {
				fprintf(stderr, "스레드 생성 실패\n");
				return 1;
			}
			
			pthread_join(tid1, NULL);
			turn = 1;
		} else {
			// 보내는 스레드가 실행되기 전에 mutex 잠금
			pthread_mutex_lock(&mutex);

			int result = pthread_create(&tid2, NULL, recvData, NULL);
			if (result != 0) {
				fprintf(stderr, "스레드 생성 실패\n");
				return 1;
    	}

			pthread_join(tid2, NULL);
			turn = 0;
		}
	}

	// 소켓 닫기
	close(sock);
	return 0;
}

void* sendData(void* arg) {
	// 데이터 입력
	printf("\nclient : ");
	if (fgets(buf, BUFSIZE + 1, stdin) == NULL)
		return NULL;

	// '\n' 문자 제거
	len = (int)strlen(buf);
	if (buf[len - 1] == '\n')
		buf[len - 1] = '\0';
	if (strlen(buf) == 0)
		return NULL;

	// 데이터 보내기
	retval = sendto(sock, buf, (int)strlen(buf), 0,
		(struct sockaddr *)&serveraddr, sizeof(serveraddr));
	if (retval == SOCKET_ERROR) {
		err_display("sendto()");
		return NULL;
	}

	// 버퍼 초기화
	memset(buf, 0, sizeof(buf));

	// mutex 잠금 해제
	pthread_mutex_unlock(&mutex);


	return 0;
}

void* recvData(void* arg) {
	// 데이터 받기
	addrlen = sizeof(peeraddr);
	retval = recvfrom(sock, buf, BUFSIZE, 0,
		(struct sockaddr *)&peeraddr, &addrlen);
	if (retval == SOCKET_ERROR) {
		err_display("recvfrom()");
		return NULL;
	}

	// 송신자의 주소 체크
	if (memcmp(&peeraddr, &serveraddr, sizeof(peeraddr))) {
		printf("[오류] 잘못된 데이터입니다!\n");
		return NULL;
	}

	// 받은 데이터 출력
	printf("\nserver : %s\n", buf);

	// 버퍼 초기화
	memset(buf, 0, sizeof(buf));

	// mutex 잠금 해제
	pthread_mutex_unlock(&mutex);
	
	return 0;
}