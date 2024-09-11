#include "./Common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#define MAX_BUFFER_SIZE 50
#define PACKET_SIZE 9

typedef struct
{
	char *buffer;
	size_t size;
	size_t length;
} StringBuffer;

void initStringBuffer(StringBuffer *sb, size_t initial_size)
{
	sb->buffer = (char *)malloc(initial_size);
	if (sb->buffer == NULL)
	{
		fprintf(stderr, "Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}
	sb->buffer[0] = '\0';
	sb->size = initial_size;
	sb->length = 0;
}

void appendStringBuffer(StringBuffer *sb, const char *str)
{
	size_t str_len = strlen(str);
	if (sb->length + str_len + 1 > sb->size)
	{
		size_t new_size = sb->size;
		while (new_size <= sb->length + str_len)
		{
			new_size *= 2;
		}
		char *new_buffer = (char *)realloc(sb->buffer, new_size);
		if (new_buffer == NULL)
		{
			fprintf(stderr, "Memory reallocation failed\n");
			free(sb->buffer);
			exit(EXIT_FAILURE);
		}
		sb->buffer = new_buffer;
		sb->size = new_size;
	}
	strcat(sb->buffer, str);
	sb->length += str_len;
}

void freeStringBuffer(StringBuffer *sb)
{
	free(sb->buffer);
	sb->buffer = NULL;
	sb->size = 0;
	sb->length = 0;
}

typedef struct Packet
{
	char payload[6]; // UTF-8 최대 4 바이트 + 공백 1 바이트 + NULL 2 바이트
	u_int16_t checksum;
	uint8_t packet_number;
} Packet;

typedef struct NODE
{
	struct NODE *next;
	struct Packet packet;
	struct timeval timeout; // 노드에 시간 기록 추가
} Node;

typedef struct LISTMARK
{
	Node *head;
	Node *tail;
	int size;
	int byte_size;
	pthread_mutex_t lock; // 버퍼 접근 동기화를 위한 뮤텍스
} ListMark;

ListMark *Init_List(void)
{
	ListMark *ls = (ListMark *)malloc(sizeof(ListMark));
	ls->head = NULL;
	ls->tail = NULL;
	ls->size = 0;
	ls->byte_size = 0;
	pthread_mutex_init(&ls->lock, NULL); // 뮤텍스 초기화
	return ls;
}

int Add_Last(ListMark *ls, Packet data)
{
	if (ls->byte_size + PACKET_SIZE > MAX_BUFFER_SIZE)
	{
		printf("Buffer overflow. Dropping packet.\n");
		return 0; // false
	}

	Node *tmp = (Node *)malloc(sizeof(Node));
	tmp->packet = data;
	tmp->next = NULL;

	if (ls->head == NULL)
	{
		ls->head = tmp;
		ls->tail = tmp;
		tmp->next = tmp; // 순환 구조를 유지
	}
	else
	{
		tmp->next = ls->head;
		ls->tail->next = tmp;
		ls->tail = tmp;
	}

	ls->size++;
	ls->byte_size += PACKET_SIZE;
	return 1; // true
}

// 타임아웃 세팅 추간
void Add_Last_Sender(ListMark *ls, Packet data)
{
	Node *tmp = (Node *)malloc(sizeof(Node));
	tmp->packet = data;
	tmp->next = NULL;

	struct timeval now;
	gettimeofday(&now, NULL);

	tmp->timeout.tv_sec = now.tv_sec;
	tmp->timeout.tv_usec = now.tv_usec + 500000;
	if (tmp->timeout.tv_usec >= 1000000)
	{
		tmp->timeout.tv_sec += 1;
		tmp->timeout.tv_usec -= 1;
	}

	if (ls->head == NULL)
	{
		ls->head = tmp;
		ls->tail = tmp;
		tmp->next = tmp; // 순환 구조를 유지
	}
	else
	{
		tmp->next = ls->head;
		ls->tail->next = tmp;
		ls->tail = tmp;
	}

	ls->size++;
}

Packet Delete_By_PacketNumber(ListMark *ls, int packet_number)
{
	Packet removedData = {0};

	if (ls->head == NULL)
	{
		puts("List is empty");
		return removedData;
	}

	Node *cur = ls->head;
	Node *prev = NULL;

	do
	{
		if (cur->packet.packet_number == packet_number)
		{
			if (prev == NULL)
			{
				// 삭제할 노드가 head인 경우
				if (ls->head == ls->tail)
				{
					ls->head = NULL;
					ls->tail = NULL;
				}
				else
				{
					ls->head = cur->next;
					ls->tail->next = ls->head;
				}
			}
			else
			{
				// 삭제할 노드가 head가 아닌 경우
				prev->next = cur->next;
				if (cur == ls->tail)
				{
					ls->tail = prev;
				}
			}

			removedData = cur->packet;
			free(cur);
			ls->size--;
			ls->byte_size -= PACKET_SIZE;
			return removedData;
		}

		prev = cur;
		cur = cur->next;
	} while (cur != ls->head);

	return removedData;
}

Packet Delete_First(ListMark *ls)
{
	Packet removedData = {0};

	if (ls->head == NULL)
	{
		puts("List is empty");
		return removedData;
	}

	Node *tmp = ls->head;
	removedData = tmp->packet;

	if (ls->head == ls->tail)
	{
		ls->head = NULL;
		ls->tail = NULL;
	}
	else
	{
		ls->head = tmp->next;
		ls->tail->next = ls->head;
	}

	free(tmp);
	ls->size--;
	ls->byte_size -= PACKET_SIZE;
	return removedData;
}

void Reset_List(ListMark *ls)
{
	Node *cur = ls->head;
	Node *tmp;
	while (ls->size > 0)
	{
		tmp = cur->next;
		free(cur);
		cur = tmp;
		ls->size--;
	}
	ls->head = NULL;
	ls->tail = NULL;
}

Node *get_Pos(ListMark *ls, int pos)
{
	if (pos <= 0 || pos > ls->size)
	{
		printf("Invalid position: %d. Position must be between 1 and %d.\n", pos, ls->size);
		return NULL;
	}

	Node *current = ls->head;
	for (int i = 1; i < pos; i++)
	{
		current = current->next;
	}

	return current;
}

void Print_List(ListMark *ls)
{
	if (ls->head == NULL)
	{
		puts("List is empty.");
		return;
	}
	Node *cur = ls->head;
	int pos = 1;
	do
	{
		printf("Node %d: Payload: %.*s, Checksum: %u, Packet Number: %d\n",
			   pos, 6, cur->packet.payload, cur->packet.checksum, cur->packet.packet_number);
		cur = cur->next;
		pos++;
	} while (cur != ls->head);
}

int Get_Length(ListMark *ls)
{
	return ls->size;
}

void Clear_List(ListMark *ls)
{
	Reset_List(ls);
	ls->head = NULL;
	ls->tail = NULL;
	ls->size = 0;
	ls->byte_size = 0;
}

#define SERVERPORT 9001

#define DATA_SIZE 100
#define WINDOW_SIZE 4

typedef struct ack_packet
{
	u_int16_t ack;
} ack_packet;

int recv_base = 0;
Packet *packets;
int packet_capacity = 10;
int packet_count = 0;
SOCKET client_sock;
ListMark *list;

uint16_t calculate_checksum(Packet *packet)
{
	uint32_t sum = 0;
	char *ptr = packet->payload;

	for (int i = 0; i < sizeof(packet->payload); i++)
	{
		sum += *ptr++;
	}
	while (sum >> 16)
	{
		sum = (sum & 0xFFFF) + (sum >> 16);
	}

	return (uint16_t)sum;
}

void ReadFromBuffer(ListMark *ls, Packet p)
{
	pthread_mutex_lock(&ls->lock); // 버퍼 접근 잠금
	if (packet_count == packet_capacity)
	{
		// 패킷 배열 용량 증가
		packet_capacity *= 2;
		Packet *new_packets = (Packet *)realloc(packets, packet_capacity * sizeof(Packet));
		if (new_packets == NULL)
		{
			perror("Failed to reallocate memory for packets");
			free(packets);
			return;
		}
		packets = new_packets;
	}

	packets[packet_count] = Delete_By_PacketNumber(ls, p.packet_number);
	struct timeval now;
	gettimeofday(&now, NULL);
	// printf("delete from buffer : %d, Time: %.5lf seconds\n",
	//    p.packet_number, (now.tv_sec + now.tv_usec / 1000000.0));
	packet_count++;

	pthread_mutex_unlock(&ls->lock); // 버퍼 접근 해제
}

// 그냥 오면 일단 버퍼로 보냄
void *recvThread(void *arg)
{
	Packet pkt;
	int retval;
	while (1)
	{
		if ((retval = recv(client_sock, &pkt, sizeof(pkt), 0)) > 0)
		{
			struct timeval now;
			gettimeofday(&now, NULL);
			Add_Last(list, pkt); // 함수 내부에 버퍼 사이즈를 넘기면 못 받도록 구현
		}
	}
}

StringBuffer sb;

int main(int argc, char *argv[])
{
	initStringBuffer(&sb, 100);

	list = Init_List();
	int ack_num = 0;
	packets = (Packet *)malloc(packet_capacity * sizeof(Packet));

	int retval;

	// 소켓 생성
	SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, 0);
	if (listen_sock == INVALID_SOCKET)
		err_quit("socket()");

	// bind()
	struct sockaddr_in serveraddr;
	memset(&serveraddr, 0, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
	serveraddr.sin_port = htons(SERVERPORT);
	retval = bind(listen_sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr));
	if (retval == SOCKET_ERROR)
		err_quit("bind()");

	// listen()
	retval = listen(listen_sock, SOMAXCONN);
	if (retval == SOCKET_ERROR)
		err_quit("listen()");

	// 데이터 통신에 사용할 변수
	struct sockaddr_in clientaddr;
	socklen_t addrlen;
	char addr[INET_ADDRSTRLEN];

	while (1)
	{
		// 클라이언트 연결 수락
		addrlen = sizeof(clientaddr);
		client_sock = accept(listen_sock, (struct sockaddr *)&clientaddr, &addrlen);
		if (client_sock == INVALID_SOCKET)
		{
			err_display("accept()");
			break;
		}

		inet_ntop(AF_INET, &clientaddr.sin_addr, addr, sizeof(addr));
		printf("\n[TCP 서버] 클라이언트 접속: IP 주소=%s, 포트 번호=%d\n",
			   addr, ntohs(clientaddr.sin_port));

		// 연결이 되면 루프를 벗어남
		break;
	}

	pthread_t tid1;
	if (pthread_create(&tid1, NULL, recvThread, NULL) != 0)
	{
		fprintf(stderr, "thread create error\n");
		exit(1);
	}

	while (1)
	{

		// 버퍼에서 읽을 데이터가 있으면
		if (list->size > 0)
		{
			Node node = *get_Pos(list, 1);
			struct Packet pkt = node.packet;
			usleep(100000);

			if (strcmp(pkt.payload, "end") == 0)
			{
				ack_packet ack;
				ack.ack = 0;
				send(client_sock, &ack, sizeof(ack_packet), 0);
				break;
			}

			uint16_t checksum = calculate_checksum(&pkt);
			if ((checksum | pkt.checksum) != 0xFFFF) // OR 연산을 통해 모든 비트가 1인지 확인
			{
				printf("checksum is different result\n");
				// 옜날에 보낸 ack 다시 보냄
				continue;
			}

			if (pkt.packet_number == recv_base)
			{
				char str[6];
				sprintf(str, "%.*s", 6, pkt.payload);
				appendStringBuffer(&sb, str);
				ack_num += sizeof(Packet) - 1;
				struct ack_packet ack;
				ack.ack = ack_num;
				if (ack.ack > 297)
				{
					printf("%s\n", pkt.payload);
				}
				retval = send(client_sock, &ack, sizeof(ack_packet), 0);
				recv_base++;
				printf("packet%d is received and there is no error. (%.*s) (Ack = %d) is transmitted \n", pkt.packet_number, 6, pkt.payload, ack.ack);
				ReadFromBuffer(list, pkt);
				continue;
			}

			// 패킷 순서가 어긋났을 때
			if (pkt.packet_number != recv_base)
			{
				// Print_List(list);
				struct ack_packet ack;
				ack.ack = ack_num;
				retval = send(client_sock, &ack, sizeof(ack_packet), 0);
				if (retval == -1)
				{
					if (errno == ECONNRESET || errno == EPIPE)
					{
						// 연결이 끊겼을 때 발생하는 에러
						fprintf(stderr, "Error: Connection closed by peer. Exiting...\n");
						break;
					}
					else
					{
						// 기타 에러 처리
						perror("send");
						break;
					}
				}
				printf("packet%d is received and dropped. (%.*s) (Ack = %d)is transmitted\n", pkt.packet_number, 6, pkt.payload, ack.ack);
				pthread_mutex_lock(&list->lock); // 버퍼 접근 잠금
				Delete_By_PacketNumber(list, pkt.packet_number);
				pthread_mutex_unlock(&list->lock);
				continue;
			}
		}
	}
	printf("%s\n", sb.buffer);
	close(client_sock);
	printf("[TCP 서버] 클라이언트 종료: IP 주소=%s, 포트 번호=%d\n", addr, ntohs(clientaddr.sin_port));

	FILE *file;

	file = fopen("output.txt", "w");

	if (file == NULL)
	{
		perror("Error opening file");
		return 1;
	}

	fprintf(file, "%s\n", sb.buffer);

	fclose(file);
	return 0;
}