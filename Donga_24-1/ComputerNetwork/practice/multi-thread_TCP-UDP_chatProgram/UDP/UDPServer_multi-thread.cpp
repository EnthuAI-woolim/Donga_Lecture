#include "../Common.h"

#define SERVERPORT 9000
#define BUFSIZE    512

void* sendData(void* arg);
void* recvData(void* arg);

// 데이터 통신에 사용할 변수
int retval;
SOCKET sock;
struct sockaddr_in clientaddr;
socklen_t addrlen;
char buf[BUFSIZE + 1];
int len;
int result;
int turn = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; // mutex 초기화

int main(int argc, char *argv[])
{
	pthread_t tid1, tid2;

	// 소켓 생성
	sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock == INVALID_SOCKET) err_quit("socket()");

	// bind()
	struct sockaddr_in serveraddr;
	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
	serveraddr.sin_port = htons(SERVERPORT);
	retval = bind(sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
	if (retval == SOCKET_ERROR) err_quit("bind()");



	// 클라이언트와 데이터 통신
	while (1) {
		if (turn == 0) {
			// 보내는 스레드가 실행되기 전에 mutex 잠금
			pthread_mutex_lock(&mutex);

			int result = pthread_create(&tid1, NULL, recvData, NULL);
			if (result != 0) {
				fprintf(stderr, "스레드 생성 실패\n");
				return 1;
			}

			pthread_join(tid1, NULL);
			turn = 1;
		} else {
			// 보내는 스레드가 실행되기 전에 mutex 잠금
			pthread_mutex_lock(&mutex);

			int result = pthread_create(&tid2, NULL, sendData, NULL);
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

void* recvData(void* arg) {
	// 데이터 받기
	addrlen = sizeof(clientaddr);
	retval = recvfrom(sock, buf, BUFSIZE, 0,
		(struct sockaddr *)&clientaddr, &addrlen);
	if (retval == SOCKET_ERROR) {
		err_display("recvfrom()");
		return NULL;
	}
	// 받은 데이터 출력
	printf("\nclient : %s\n", buf);

	// 버퍼 초기화
	memset(buf, 0, sizeof(buf));

	// mutex 잠금 해제
	pthread_mutex_unlock(&mutex);

	return 0;
}

void* sendData(void* arg) {
	// 데이터 입력
	printf("\nserver : ");
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
		(struct sockaddr *)&clientaddr, sizeof(clientaddr));
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