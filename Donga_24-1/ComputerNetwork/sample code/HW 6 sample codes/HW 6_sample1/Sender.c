// sender.c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/time.h>
#include <locale.h>

#define MAX_BUFFER_SIZE 50
#define PACKET_SIZE 9

typedef struct Packet
{
    char payload[6]; 
    u_int16_t checksum;
    uint8_t packet_number;
} Packet;

typedef struct NODE
{
    struct NODE *next;
    struct Packet packet;
    struct timeval timeout; 
} Node;

typedef struct LISTMARK
{
    Node *head;
    Node *tail;
    int size;
    int byte_size;
} ListMark;

ListMark *Init_List(void)
{
    ListMark *ls = (ListMark *)malloc(sizeof(ListMark));
    ls->head = NULL;
    ls->tail = NULL;
    ls->size = 0;
    ls->byte_size = 0;
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
        tmp->next = tmp; // ��ȯ ������ ����
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

// Ÿ�Ӿƿ� ���� �߰�
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
        tmp->timeout.tv_usec -= 1000000;
    }

    if (ls->head == NULL)
    {
        ls->head = tmp;
        ls->tail = tmp;
        tmp->next = tmp; // ��ȯ ������ ����
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
                // ������ ��尡 head�� ���
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
        printf("Node %d: Payload: %02X %02X %02X %02X, Checksum: %04X, Packet Number: %d\n",
               pos, cur->packet.payload[0], cur->packet.payload[1], cur->packet.payload[2], cur->packet.payload[3], cur->packet.checksum, cur->packet.packet_number);
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

#define MAX_SEQ 8
#define WINDOW_SIZE 4
#define TIMEOUT 500000
#define SERVER_PORT 9001
#define SERVER_IP "127.0.0.1"
#define DATA_SIZE 100

typedef struct ack_packet
{
    u_int16_t ack;
} ack_packet;

struct timer
{
    struct timeval end_time;
};

int sock;
struct sockaddr_in server_addr;
Packet *packets;
int packet_count = 0;
int packet_capacity = 10; // �ʱ� �뷮
int *isAcked;

int next_pkt = 0;
int send_base = 0;
ack_packet acks[WINDOW_SIZE];
uint16_t ack_size = 0;
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

    // üũ���� ���� 1�� ����
    return (uint16_t)~sum;
}

// timeval ����ü �� �Լ�
int timeval_cmp(struct timeval *a, struct timeval *b)
{
    if (a->tv_sec > b->tv_sec)
    {
        return 1; 
    }
    else if (a->tv_sec < b->tv_sec)
    {
        return -1; 
    }
    else
    {
        if (a->tv_usec > b->tv_usec)
        {
            return 1;
        }
        else if (a->tv_usec < b->tv_usec)
        {
            return -1; 
        }
        else
        {
            return 0; 
        }
    }
}

int check_timeout(ListMark *tq)
{
    if (tq->size == 0)
    {
        return;
    }

    Node *node = tq->head;
    struct timeval timeout = node->timeout;

    struct timeval now;
    gettimeofday(&now, NULL);

    return (timeval_cmp(&now, &timeout) > 0);
}

int makePackets()
{
    FILE *file = fopen("text.txt", "rb"); 
    if (!file)
    {
        perror("Failed to open file");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc(file_size);
    if (!buffer)
    {
        fclose(file);
        perror("Failed to allocate memory for buffer");
        return 1;
    }
    fread(buffer, 1, file_size, file);

    packets = (Packet *)malloc(packet_capacity * sizeof(Packet));
    if (!packets)
    {
        free(buffer);
        fclose(file);
        perror("Failed to allocate memory for packets");
        return 1;
    }

    for (int i = 0; i < file_size;)
    {
        int bytes = 0;
        int chars_to_read = 2; // �� ���ھ�

        while (chars_to_read > 0 && i + bytes < file_size)
        {
            unsigned char ch = buffer[i + bytes];
            if (ch >= 0xC0)
            { // ��Ƽ����Ʈ ����
                if (ch < 0xE0)
                    bytes += 2;
                else if (ch < 0xF0)
                    bytes += 3;
                else if (ch < 0xF8)
                    bytes += 4;
            }
            else
            {
                bytes += 1; // ASCII
            }
            chars_to_read--;
        }

        if (packet_count == packet_capacity)
        {
            // ��Ŷ �迭 �뷮 ����
            packet_capacity *= 2;
            Packet *new_packets = (Packet *)realloc(packets, packet_capacity * sizeof(Packet));
            if (new_packets == NULL)
            {
                perror("Failed to reallocate memory for packets");
                free(packets);
                free(buffer);
                fclose(file);
                return 1;
            }
            packets = new_packets;
        }

        Packet packet = {0};
        memcpy(packet.payload, buffer + i, bytes);
        sprintf(packet.payload, "%.*s", 6, packet.payload);
        packet.packet_number = packet_count;

        packets[packet_count++] = packet;

        i += bytes; // ���� ���ڷ� �̵�
    }

    free(buffer);
    fclose(file);
    return packet_count;
}

int checkDuplicate(int sendBase, int ack)
{
    for (int i = 0; i < sendBase; i++)
    {
        if (isAcked[i] == ack)
        {
            return 1;
        }
    }
    return 0;
}

void shift_left(ack_packet acks[], uint16_t *ack_count)
{
    if (*ack_count > 0)
    {
        for (uint16_t i = 1; i < *ack_count; i++)
        {
            acks[i - 1] = acks[i];
        }
        (*ack_count)--;
    }
}

void *recvThread(void *arg)
{
    ack_packet ack;
    while (1)
    {
        if (recv(sock, &ack, sizeof(ack_packet), 0) > 0)
        {
            if (checkDuplicate(send_base, ack.ack))
            {
                printf("(Ack = %d) is received and ignored\n", ack.ack);
                continue;
            }
            // ���� ���ϱ� ack�� �ϰ� �������� ����
            if (next_pkt >= packet_count)
            {
                printf("(Ack = %d) is received\n", ack.ack);
                Packet rPacket = Delete_First(list);
                isAcked[send_base] = ack.ack;
                send_base++;
                continue;
            }
            int pkt_num = ack.ack / (sizeof(Packet) - 1) - 1;
            Delete_By_PacketNumber(list, pkt_num);
            acks[ack_size].ack = ack.ack;
            ack_size++;
            isAcked[send_base] = ack.ack;
            send_base++;
        }
    }
}

int main()
{
    list = Init_List();

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        perror("Connect failed");
        exit(EXIT_FAILURE);
    }

    int flag = fcntl(sock, F_GETFL, 0);
    if (flag == -1)
    {
        perror("fcntl get");
        close(sock);
        return EXIT_FAILURE;
    }

    flag |= O_NONBLOCK;
    if (fcntl(sock, F_SETFL, flag) == -1)
    {
        perror("fcntl set");
        close(sock);
        return EXIT_FAILURE;
    }

    int pkt_count = makePackets();
    isAcked = (int *)malloc(packet_count * sizeof(int)); // ack�� ���� Ȯ���ϱ� ���� �迭
    memset(isAcked, 0, packet_count * sizeof(int));

    // pthread_t tid1;
    // if (pthread_create(&tid1, NULL, recvThread, NULL) != 0)
    // {
    //     fprintf(stderr, "thread create error\n");
    //     exit(1);
    // }
    int c = 0;
    while (send_base < pkt_count)
    {

        // Ÿ�Ӿƿ� Ȯ���ϴ� ���ǹ�
        if (list->size > 0)
        {
            if (check_timeout(list))
            {
                Node n = *get_Pos(list, 1);
                Packet p = n.packet;
                printf("Packet%d is timeout\n", p.packet_number);

                next_pkt = p.packet_number;
                Clear_List(list);
            }
        }

        if (next_pkt >= send_base + WINDOW_SIZE)
        {
            continue;
        }

        int count = 0;
        for (int i = next_pkt; i < send_base + WINDOW_SIZE; i++)
        {
            if (next_pkt >= pkt_count)
                break;

            Packet p = packets[next_pkt];
            sprintf(p.payload, "%.*s", 6, p.payload);
            p.checksum = calculate_checksum(&p);
            // ���� ���� �� ���� ��
            if (ack_size > 0)
            {

                printf("(Ack = %d) is received and packet%d is transmitted (%.*s)\n", acks[0].ack, p.packet_number, 6, p.payload);
                shift_left(acks, &ack_size);
            }
            else
            {
                printf("packet%d is transmitted. (%.*s)\n", p.packet_number, 6, p.payload);
            }
            send(sock, &p, sizeof(Packet), 0);
            Add_Last_Sender(list, p); // ���������� Ÿ�Ӿƿ� ���� ������
            next_pkt++;
            usleep(50000);
            count++;
        }

        ack_packet ack;
        for (int i = 0; i <= count; i++)
        {

            if (recv(sock, &ack, sizeof(ack_packet), 0) > 0)
            {
                if (checkDuplicate(send_base, ack.ack))
                {
                    printf("(Ack = %d) is received and ignored\n", ack.ack);
                    continue;
                }
                // ���� ���ϱ� ack�� �ϰ� �������� ����
                if (next_pkt >= packet_count)
                {
                    printf("(Ack = %d) is received\n", ack.ack);
                    Packet rPacket = Delete_First(list);
                    isAcked[send_base] = ack.ack;
                    send_base++;
                    continue;
                }
                int pkt_num = ack.ack / (sizeof(Packet) - 1) - 1;
                Delete_By_PacketNumber(list, pkt_num);
                acks[ack_size].ack = ack.ack;
                ack_size++;
                isAcked[send_base] = ack.ack;
                send_base++;
            }
        }
    }

    Packet end;
    sprintf(end.payload, "%s", "end");
    send(sock, &end, sizeof(Packet), 0);
    while ((1))
    {
        ack_packet ack;
        if (recv(sock, &ack, sizeof(ack_packet), 0) > 0)
        {
            if (ack.ack == 0)
                break;
        }
    }

    close(sock);
}
