#include "Common.h"

#define SERVERPORT 9001
#define BUFSIZE 512

int isLeap(int year) {
    if (((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0)) return 1;
	else return 0;
}

int getNumberOfDays(int year, int month) {
	if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12) return 31;
	if (month == 4 || month == 6 || month == 9 || month == 11) return 30;
	if (month == 2) {
		if (isLeap(year) == 1) return 29;
		else return 28;
	}
	return 0;
}

int make_callendar(int year, int month) {
    int start1800 = 3;

    int total = 0;
    for (int i = 1800; i < year; i++) {
        if (isLeap(i)) total += 366;
        else total += 365;
    }
    for (int i = 1; i < month; i++) {
        total += getNumberOfDays(year, i);
    }

    return (start1800 + total) % 7;
}

char* printMonth(int year, int month) {
	static char buffer[BUFSIZE] = "";
    int numberofdays = getNumberOfDays(year, month);
    int startday = make_callendar(year, month);
	
	char s[128];
	sprintf(s, "연도: %d\n월  : %d\n  SUN MON THU WED THU FRI SAT\n", year, month);
	strcat(buffer, s);

    for (int i = 0; i < startday; i++) {
		strcat(buffer, "    ");
	}
	for (int i = 1; i <= numberofdays; i++) {
		char s[5];
		sprintf(s, "%4d", i);
		strcat(buffer, s);
		if ((i + startday) % 7 == 0) {
			strcat(buffer, "\n");
		}
	}
	strcat(buffer, "\n");

	return buffer;
}

int main(int argc, char *argv[]) {
    int retval;

    SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, 0);
	if (listen_sock == INVALID_SOCKET) err_quit("socket()");

    struct sockaddr_in serveraddr;
	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
	serveraddr.sin_port = htons(SERVERPORT);
	retval = bind(listen_sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
	if (retval == SOCKET_ERROR) err_quit("bind()");

    retval = listen(listen_sock, SOMAXCONN);
	if (retval == SOCKET_ERROR) err_quit("listen()");

    SOCKET client_sock;
	struct sockaddr_in clientaddr;
	socklen_t addrlen;
	char buf[BUFSIZE + 1];

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

		// 클라이언트와 데이터 통신
		while (1) {
			// 데이터 받기
			retval = recv(client_sock, buf, BUFSIZE, 0);
			if (retval == SOCKET_ERROR) {
				err_display("recv()");
				break;
			}
			else if (retval == 0)
				break;
			buf[retval+1] = '\0';
			
            int ym[2] = {0, 0};
			char *ptr = strtok(buf, ".");
			
			ym[0] = atoi(ptr);
			ptr = strtok(NULL, ".");
			ym[1] = atoi(ptr);
			char *buffer = printMonth(ym[0], ym[1]);
			printf("%s\n\n", buffer);

			// 데이터 보내기
			retval = send(client_sock, buffer, BUFSIZE, 0);
			if (retval == SOCKET_ERROR) {
				err_display("send()");
				break;
			}
			*buffer = NULL;
		}

		// 소켓 닫기
		close(client_sock);
	}

	// 소켓 닫기
	close(listen_sock);
	return 0;
}