#include "../Common.h"
#include "./FR_receiver.h"


int main(int argc, char *argv[])
{
	
	
	// 명령행 인수가 있으면 IP 주소로 사용
	if (argc > 1) SERVERIP = argv[1];

	// 소켓 생성
	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock == INVALID_SOCKET) err_quit("socket()");

	
	// connect()
	struct sockaddr_in serveraddr;
	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	inet_pton(AF_INET, SERVERIP, &serveraddr.sin_addr);
	serveraddr.sin_port = htons(SERVERPORT);
	retval = connect(sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
	if (retval == SOCKET_ERROR) err_quit("connect()");

	
	// 저장할 packet 정의
	Packet packet[PACKET_COUNT];
	setPacket(packet);

	// 서버와 데이터 통신
	while (1) {

		recvData(packet);

		unsigned short binary_data = generate_binary_and_checksum(&recvBuf);
		unsigned short total_sum = recvBuf.checksum + binary_data;
		if (check_all_ones(total_sum)) {

            if (recvBuf.id == expect_pkt) {
                sendData(packet);

                if (retransmission == 1) break;
            } else {
                expect_pkt = packet[expect_pkt - 1].id;	// 다음 받을 올바른 패킷번호 설정
                sendData(packet);
                retransmission = 1;
            }

        } else {
            printf("checksum 오류입니다.\n");
            break;
        }

	}
	// 소켓 닫기
	close(sock);
	return 0;
}