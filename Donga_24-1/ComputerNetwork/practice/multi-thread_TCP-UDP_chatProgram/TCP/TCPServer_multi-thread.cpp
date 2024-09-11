#include "../Common.h"

#define SERVERPORT 9000
#define BUFSIZE    512

void* sendData(void* arg);
void* recvData(void* arg);

// 데이터 통신에 사용할 변수
int retval;
SOCKET client_sock;
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

		// 접속한 클라이언트 정보 출력
		char addr[INET_ADDRSTRLEN];
		inet_ntop(AF_INET, &clientaddr.sin_addr, addr, sizeof(addr));
		printf("\n[TCP 서버] 클라이언트 접속: IP 주소=%s, 포트 번호=%d\n",
			addr, ntohs(clientaddr.sin_port));

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
		close(client_sock);
		printf("[TCP 서버] 클라이언트 종료: IP 주소=%s, 포트 번호=%d\n",
			addr, ntohs(clientaddr.sin_port));
	}

	// 소켓 닫기
	close(listen_sock);
	return 0;
}

void* recvData(void* arg) {
	// 데이터 받기
	retval = recv(client_sock, buf, BUFSIZE, 0);
	if (retval == SOCKET_ERROR) {
		err_display("recv()");
	}
	else if (retval == 0)
		return NULL;

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
	retval = send(client_sock, buf, (int)strlen(buf), 0);
	if (retval == SOCKET_ERROR) {
		err_display("send()");
	}

	// 버퍼 초기화
	memset(buf, 0, sizeof(buf));

	// mutex 잠금 해제
	pthread_mutex_unlock(&mutex);

	return 0;
}