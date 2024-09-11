#include "Common.h"

#define SERVERPORT 9000
#define BUFSIZE    512

void* send_thread(void* arg) {
    SOCKET client_sock = *((SOCKET*)arg);
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
        int retval = send(client_sock, buf, (int)strlen(buf), 0);
        if (retval == SOCKET_ERROR) {
            err_display("send()");
            break;
        }
    }
    pthread_exit(NULL);
}

void* receive_thread(void* arg) {
    SOCKET client_sock = *((SOCKET*)arg);
    char buf[BUFSIZE + 1];

    while (1) {
        // 데이터 받기
        int retval = recv(client_sock, buf, BUFSIZE, 0);
        if (retval == SOCKET_ERROR) {
            err_display("recv()");
            break;
        }
        else if (retval == 0)
            break;

        // 받은 데이터 출력
        buf[retval] = '\0';
        printf("client: %s\n", buf);
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[])
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
    retval = bind(listen_sock, (struct sockaddr*)&serveraddr, sizeof(serveraddr));
    if (retval == SOCKET_ERROR) err_quit("bind()");

    // listen()
    retval = listen(listen_sock, SOMAXCONN);
    if (retval == SOCKET_ERROR) err_quit("listen()");

    // 데이터 통신에 사용할 변수
    SOCKET client_sock;
    struct sockaddr_in clientaddr;
    socklen_t addrlen;

    while (1) {
        // accept()
        addrlen = sizeof(clientaddr);
        client_sock = accept(listen_sock, (struct sockaddr*)&clientaddr, &addrlen);
        if (client_sock == INVALID_SOCKET) {
            err_display("accept()");
            break;
        }
        printf("\n** chat **\n");

        // 스레드 생성 및 실행
        pthread_t receive_tid, send_tid;
        if(pthread_create(&receive_tid, NULL, receive_thread, (void*)&client_sock)){
		printf("thread create error\n");
		exit(1);
    	}
        if(pthread_create(&send_tid, NULL, send_thread, (void*)&client_sock)){
        	printf("thread create error\n");
		exit(1);
	}
        
        pthread_join(receive_tid, NULL);
        pthread_join(send_tid, NULL);
        
        // 소켓 닫기
        close(client_sock);
    }

    // 소켓 닫기
    close(listen_sock);
    return 0;
}

