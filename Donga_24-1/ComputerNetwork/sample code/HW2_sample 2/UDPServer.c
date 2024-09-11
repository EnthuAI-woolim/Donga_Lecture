#include "Common.h"

#define SERVERPORT 9000
#define BUFSIZE    3500

int main(int argc, char *argv[])
{
	int retval;

	// 소켓 생성
	SOCKET sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock == INVALID_SOCKET) err_quit("socket()");

	// bind()
	struct sockaddr_in serveraddr;
	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
	serveraddr.sin_port = htons(SERVERPORT);
	retval = bind(sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
	if (retval == SOCKET_ERROR) err_quit("bind()");

	// 데이터 통신에 사용할 변수
	struct sockaddr_in clientaddr;
	socklen_t addrlen;
	char buf[BUFSIZE + 1];

	// 클라이언트와 데이터 통신
	while (1) {
		// 데이터 받기
		addrlen = sizeof(clientaddr);
		retval = recvfrom(sock, buf, BUFSIZE, 0,
			(struct sockaddr *)&clientaddr, &addrlen);
		if (retval == SOCKET_ERROR) {
			err_display("recvfrom()");
			break;
		}
		else
			printf(">> The server received a request from a client.\n");
		
		// 파일 열기, 읽기
		char filename[20];
		buf[retval] = '\0';
		sscanf(buf, "request \"%19[^\"]", filename);
		FILE* file = fopen(filename, "r");

		if (file == NULL) {
			printf("파일 이름 : %s\n",filename);
			printf("파일 열기 실패\n");
			break;
		}
	
		retval = fread(buf, 1, BUFSIZE, file);
		
		// 데이터 보내기
		retval = sendto(sock, buf, retval, 0,
			(struct sockaddr *)&clientaddr, sizeof(clientaddr));
		if (retval == SOCKET_ERROR) {
			err_display("sendto()");
			break;
		}
		else
			printf(">> The server sent \"%s\" to the client.\n\n", filename);

		fclose(file);
		}
	
	// 소켓 닫기
	close(sock);
	return 0;
}
