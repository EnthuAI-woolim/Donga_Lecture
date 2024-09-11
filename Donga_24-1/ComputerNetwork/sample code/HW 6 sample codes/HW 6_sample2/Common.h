#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <poll.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

typedef int SOCKET;
#define SOCKET_ERROR -1
#define INVALID_SOCKET -1

typedef struct Information {
    int indicator;
    int send_base;
    int nextseqnum;
    int end;
    clock_t timeout;
}Information;

void err_quit(const char *msg) {
    char *msgbuf = strerror(errno);
    printf("[%s] %s\n", msg, msgbuf);
    exit(1);
}

void err_display(const char *msg) {
    char *msgbuf = strerror(errno);
    printf("[%s] %s\n", msg, msgbuf);
}
/*
void err_display(int errcode) {
    char *msgbuf = strerror(errcode);
    printf("[오류] %s\n", msgbuf);
}
*/