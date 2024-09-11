#include "../Common.h"

#define SERVERPORT 9000
#define BUFSIZE    1024

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
  char request[20];
  char filename[20];
  FILE *file;
  long file_size;
  size_t read_len;

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

    // 받은 데이터에서 request와 "novel.txt"분리
    char *ptr_request = strstr(buf, "request");
    if(ptr_request != NULL) {
        sscanf(ptr_request, "%s \"%[^\"]\"", request, filename);
    }
		printf("The server received a %s from a client\n", request);	// 요청 수신 확인 출력

    // 파일 가져오기
		if((file = fopen(filename, "r")) == NULL) {
			printf("ERROR-cannot open %s \n", filename);
			break;
		}
    
    // 파일 크기 구하기
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // 파일 내용 읽어서 클라이언트로 전송
    while ((read_len = fread(buf, 1, BUFSIZE, file)) > 0) {

      retval = sendto(sock, buf, read_len, 0,
        (struct sockaddr *)&clientaddr, sizeof(clientaddr));
      if (retval == SOCKET_ERROR) {
        err_display("sendto()");
        break;
      }
    }
    printf("The server sent a \"%s\" to the client\n", filename);	// 송신 완료 확인 출력

    // 파일 닫기
    fclose(file);
	}

	// 소켓 닫기
	close(sock);
	return 0;
}