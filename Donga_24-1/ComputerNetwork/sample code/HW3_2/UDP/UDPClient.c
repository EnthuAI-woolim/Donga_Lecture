#include "Common.h"

char* SERVERIP = (char*)"127.0.0.1";
#define SERVERPORT 9000
#define BUFSIZE    512

struct sockaddr_in serveraddr;

// send 스레드 함수
void* send_thread(void* arg)
{
    SOCKET sock = *((SOCKET*)arg);
    int retval;
    char buf[BUFSIZE + 1];

    while (1)
    {
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
        retval = sendto(sock, buf, (int)strlen(buf), 0,
            (struct sockaddr*)&serveraddr, sizeof(serveraddr));
        if (retval == SOCKET_ERROR)
        {
            err_display("sendto()");
            break;
        }
    }

    pthread_exit(NULL);
}

// receive 스레드 함수
void* receive_thread(void* arg)
{
    SOCKET sock = *((SOCKET*)arg);
    int retval;
    char buf[BUFSIZE + 1];
    socklen_t addrlen;
    
    while (1)
    {
        // 데이터 받기
        addrlen = sizeof(serveraddr);
        retval = recvfrom(sock, buf, BUFSIZE, 0,
            (struct sockaddr*)&serveraddr, &addrlen);
        if (retval == SOCKET_ERROR)
        {
            err_display("recvfrom()");
            break;
        }

        // 받은 데이터 출력
        buf[retval] = '\0';
        printf("server: %s\n", buf);
    }

    pthread_exit(NULL);
}

int main(int argc, char* argv[])
{
    int retval;
    
    // 소켓 주소 구조체 초기화
    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    inet_pton(AF_INET, SERVERIP, &serveraddr.sin_addr);
    serveraddr.sin_port = htons(SERVERPORT);

    // 명령행 인수가 있으면 IP 주소로 사용
    if (argc > 1) SERVERIP = argv[1];

    // 소켓 생성
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock == INVALID_SOCKET) err_quit("socket()");

    printf("\n** chat **\n");

    // 스레드 변수
    pthread_t send_tid, receive_tid;

    // 스레드 생성
    pthread_create(&send_tid, NULL, send_thread, (void*)&sock);
    pthread_create(&receive_tid, NULL, receive_thread, (void*)&sock);

    // 스레드 종료 대기
    pthread_join(send_tid, NULL);
    pthread_join(receive_tid, NULL);

    // 소켓 닫기
    close(sock);
    return 0;
}

