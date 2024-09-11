#include "../Common.h"

#define SERVERPORT 9000
#define BUFSIZE    512

void calculateAndStoreCalendar(int year, int month, char *calendarData);
int isLeapYear(int year);
int getFirstWeekday(int year, int month);
int getTotalDays(int year, int month);


int main(int argc, char *argv[])
{
	int retval, year, month;

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

	// 데이터 통신에 사용할 변수
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
		printf("\n[TCP 서버] 클라이언트 접속: IP 주소=%s, 포트 번호=%d\n",
			addr, ntohs(clientaddr.sin_port));

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
			
			// buf 문자열을 "."을 기준으로 토큰으로 분리하여 연도와 월 추출
			char *token = strtok(buf, ".");
			year = atoi(token); // 문자열을 정수로 변환하여 year 변수에 저장

			// 다음 토큰으로 이동하여 월을 추출
			token = strtok(NULL, ".");
			month = atoi(token); // 문자열을 정수로 변환하여 month 변수에 저장

      // 받은 데이터 출력
      buf[retval] = '\0';
			calculateAndStoreCalendar(year, month, buf);
			printf("%s", buf);


			// 데이터 보내기
			retval = send(client_sock, buf, BUFSIZE, 0);
			if (retval == SOCKET_ERROR) {
				err_display("send()");
				break;
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

// 년도와 월에 맞는 달력 계산 및 저장 함수
void calculateAndStoreCalendar(int year, int month, char *buf) {

	sprintf(buf, "SUN MON THU WED THU FRI SAT\n");

	// 해당 월의 첫째 날의 요일을 계산
	int firstWeekday = getFirstWeekday(year, month);

	// 해당 월의 총 일 수 계산
	int totalDays = getTotalDays(year, month);

	// 달력 데이터를 문자열에 저장
	int dayCount = 1;
	for (int i = 0; i < 6; i++) { // 최대 6주
		for (int j = 0; j < 7; j++) { // 일주일은 7일
			if (i == 0 && j < firstWeekday) {
				// 첫째 주이고, 첫째 날의 요일 이전인 경우 공백으로 채우기
				sprintf(buf, "%4s    ", buf);
			} else if (dayCount <= totalDays) {
				// 해당 월의 날짜를 추가
				sprintf(buf, "%4s%3d ", buf, dayCount++);
			} else {
				// 날짜가 더 이상 없는 경우 빈칸으로 채우기
				sprintf(buf, "%4s   ", buf);
			}
		}
		sprintf(buf, "%s\n", buf); // 줄 바꿈
	}
}

// 윤년 여부를 판단하는 함수
int isLeapYear(int year) {
	return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

// 해당 월의 첫째 날의 요일을 반환하는 함수
int getFirstWeekday(int year, int month) {
	// 1년 1월 1일부터 해당 년도 이전 년도 1월 1일까지의 총 일 수 계산
	int totalDays = 0;
	for (int i = 1; i < year; i++) {
			totalDays += isLeapYear(i) ? 366 : 365;
	}

	// 해당 년도의 1월 1일부터 해당 월 1일까지의 총 일 수 계산
	int daysInMonth[] = {31, 28 + isLeapYear(year), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
	for (int i = 0; i < month - 1; i++) {
			totalDays += daysInMonth[i];
	}

	// 해당 월의 첫째 날의 요일 계산 (0: 일요일, 1: 월요일, ..., 6: 토요일)
	int firstWeekday = (totalDays + 1) % 7;

	return firstWeekday;
}

// 해당 월의 총 일 수를 반환하는 함수
int getTotalDays(int year, int month) {

	// 각 달의 일 수를 배열에 저장 (index 0은 사용하지 않음)
	int daysInMonth[] = {0, 31, 28 + isLeapYear(year), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

	// 해당 월의 총 일 수 반환
	return daysInMonth[month];
}
