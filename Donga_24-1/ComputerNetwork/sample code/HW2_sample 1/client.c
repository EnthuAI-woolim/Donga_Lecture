#include "Common.h"

char *SERVERIP = (char *)"127.0.0.1";
#define SERVERPORT 9000
#define BUFSIZE    3078

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
	struct sockaddr_in peeraddr;	// 보내온 상대방 주소
	socklen_t addrlen;				// 소켓주소 길이
	char buf[BUFSIZE + 1];			// 버퍼
	int len;

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
		printf("request “novel.txt\n");

		// 데이터 받기
		addrlen = sizeof(peeraddr);
		retval = recvfrom(sock, buf, BUFSIZE, 0,
			(struct sockaddr *)&peeraddr, &addrlen);
		if (retval == SOCKET_ERROR) {
			err_display("recvfrom()");
			break;
		}

		// 송신자의 주소 체크
		if (memcmp(&peeraddr, &serveraddr, sizeof(peeraddr))) {
			printf("[오류] 잘못된 데이터입니다!\n");
			break;
		}

		// 받은 데이터 출력
		buf[retval] = '\0';

		// 소설을 받았으면 잘 받았다고 출력하기
		if(retval > 3000) {printf("The client received “novel.txt” from the server.\n\n");}

		printf("[UDP 클라이언트] %d바이트를 받았습니다.\n", retval);
		printf("[받은 데이터] %s\n", buf);

		// 파일 열기 
		int fileOpen = open("novelOnClient.txt", O_WRONLY | O_CREAT | O_TRUNC); // 없으면 만들고, 이미 있으면 지우고 새로 쓰는 플래그 적용

		// 파일 쓰기 
		write(fileOpen,buf,retval);
		// 파일 닫기
		close(fileOpen);		
	}

	// 소켓 닫기
	close(sock);
	return 0;
}