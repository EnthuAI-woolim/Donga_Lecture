#include "Common.h"

char* SERVERIP = (char*)"127.0.0.1";
#define SERVERPORT 9000
#define BUFSIZE    512

void* send_thread(void* arg) {
    SOCKET sock = *((SOCKET*)arg);
    char buf[BUFSIZE + 1];

    while (1) {
        // 데이터 입력
        if (fgets(buf, BUFSIZE + 1, stdin) == NULL)
            break;

        // '\n' 문자 제거
        int len = (int)strlen(buf);
        if (buf[len - 1] == '\n')
            buf[len - 1] = '\0';
        if (strlen(buf) == 0)
            break;

        // 데이터 보내기
        int retval = send(sock, buf, (int)strlen(buf), 0);
        if (retval == SOCKET_ERROR) {
            err_display("send()");
            break;
        }
    }
    pthread_exit(NULL);
}

void* receive_thread(void* arg) {
    SOCKET sock = *((SOCKET*)arg);
    char buf[BUFSIZE + 1];

    while (1) {
        // 데이터 받기
        int retval = recv(sock, buf, BUFSIZE, 0);
        if (retval == SOCKET_ERROR) {
            err_display("recv()");
            break;
        }
        else if (retval == 0)
            break;

        // 받은 데이터 출력
        buf[retval] = '\0';
        printf("server: %s\n", buf);
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[])
{
    int retval;

    // 소켓 생성
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) err_quit("socket()");

    // connect()
    struct sockaddr_in serveraddr;
    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    inet_pton(AF_INET, SERVERIP, &serveraddr.sin_addr);
    serveraddr.sin_port = htons(SERVERPORT);
    retval = connect(sock, (struct sockaddr*)&serveraddr, sizeof(serveraddr));
    if (retval == SOCKET_ERROR) err_quit("connect()");
    
    printf("\n** chat **\n");
    // 스레드 생성 및 실행
    pthread_t send_tid, receive_tid;
    if(pthread_create(&send_tid, NULL, send_thread, (void*)&sock)){
	printf("thread create error\n");
	exit(1);
    }
    if(pthread_create(&receive_tid, NULL, receive_thread, (void*)&sock)){
	printf("thread create error\n");
	exit(1);
    }
    
    pthread_join(send_tid, NULL);
    pthread_join(receive_tid, NULL);

    // 소켓 닫기
    close(sock);
    return 0;
}
