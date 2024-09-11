#include "../Common.h"

char *SERVERIP = (char *)"127.0.0.1";
#define SERVERPORT 9000
#define BUFSIZE    1024

int main(int argc, char *argv[])
{
	int retval;

	// 명령행 인수가 있으면 IP 주소로 사용
	if (argc > 1) SERVERIP = argv[1];

	// 소켓 생성
	SOCKET sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock == INVALID_SOCKET) err_quit("socket()");

	// 소켓 주소 구조체 초기화
	struct sockaddr_in serveraddr;
	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	inet_pton(AF_INET, SERVERIP, &serveraddr.sin_addr);
	serveraddr.sin_port = htons(SERVERPORT);

	// 데이터 통신에 사용할 변수
	struct sockaddr_in peeraddr;
	socklen_t addrlen;
	char buf[BUFSIZE + 1];
  char request[20];
  char filename[20];
  char recv_filename[20];
  int len;
  char ch;
  FILE *file;

	// 서버와 데이터 통신
	while (1) {
		// 데이터 입력
		printf("\n[보낼 데이터] ");
		if (fgets(buf, BUFSIZE + 1, stdin) == NULL)
			break;

		// '\n' 문자 제거
		len = (int)strlen(buf);
		if (buf[len - 1] == '\n')
			buf[len - 1] = '\0';
		if (strlen(buf) == 0)
			break;

		// 데이터 보내기
		retval = sendto(sock, buf, (int)strlen(buf), 0,
			(struct sockaddr *)&serveraddr, sizeof(serveraddr));
		if (retval == SOCKET_ERROR) {
			err_display("sendto()");
			break;
		}
		printf("[UDP 클라이언트] %d바이트를 보냈습니다.\n", retval);

    // 받은 데이터에서 request와 "novel.txt"분리
    char *ptr_request = strstr(buf, "request");
    if(ptr_request != NULL) {
      sscanf(ptr_request, "%s \"%[^\"]\"", request, filename);
    }

    // 저장할 파일명 설정
    strcpy(recv_filename, "recv_");
    strcat(recv_filename, filename);

     // 파일 열기
    file = fopen(recv_filename, "w");
    if (file == NULL) {
      perror("fopen() error");
      close(sock);
      exit(1);
    }

    while (1) {
      // 데이터 받기
      addrlen = sizeof(peeraddr);
      retval = recvfrom(sock, buf, BUFSIZE, 0, 
        (struct sockaddr *)&peeraddr, &addrlen);
      if (retval == SOCKET_ERROR) {
        err_display("recvfrom()");
        break;
      } 
      // 파일에 데이터 쓰기
      fwrite(buf, 1, retval, file);

      // 받은 데이터의 크기가 버퍼 크기 미만이면 전송이 완료된 것으로 판단
			if (retval < BUFSIZE) {
				break;
			}
    }
    fclose(file);
    printf("The client received \"%s\" from the server.\n\n", filename); // 수신 완료 출력

    // 파일 읽기
		if((file = fopen(recv_filename, "r")) == NULL) {
			printf("ERROR-cannot open %s \n", recv_filename);
			break;
		}

    // 저장한 파일 화면에 출력
    while ((ch = fgetc(file)) != EOF) {
      printf("%c", ch);
    }
    fclose(file);

		// 송신자의 주소 체크
		if (memcmp(&peeraddr, &serveraddr, sizeof(peeraddr))) {
			printf("[오류] 잘못된 데이터입니다!\n");
			break;
		}
	}

	// 소켓 닫기
	close(sock);
	return 0;
}