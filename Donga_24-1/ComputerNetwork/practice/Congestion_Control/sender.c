#include <pthread.h>
#include "./Common.h"

#define SERVERPORT 9000
#define PACKETSIZE 6  // seq_num(2) + checksum(2) + payload(2)
#define PAYLOADSIZE 2
#define BUFSIZE 70
#define T 0.5
#define TIMEOUT 5

char *SERVERIP = "127.0.0.1";
int sock;
FILE* fp;
char txt[513]; // txt 파일을 읽어올 버퍼

typedef struct {
    int seq_num;
    char payload[PAYLOADSIZE + 1];
    unsigned char checksum[2];
} Packet;

char buffer[BUFSIZE + 1];
int base = 0;
int nextseqnum = 0;
int expectedACK = 0;
int duplicateACKs = 0;
int cwnd = 1;
int ssthresh = 16;
int indicator = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void calculate_checksum(Packet *pkt) {
    int sum = 0;
    for (int i = 0; i < PAYLOADSIZE; i++) {
        sum += pkt->payload[i];
    }
    sum = (sum & 0xFF) + (sum >> 8);
    pkt->checksum[0] = ~(sum & 0xFF);
    pkt->checksum[1] = '\0';
}

void *send_thread(void *arg) {
    while (1) {
        pthread_mutex_lock(&mutex);
        if (nextseqnum < base + cwnd) {
            Packet pkt;
            pkt.seq_num = nextseqnum;
            strncpy(pkt.payload, txt + nextseqnum * PAYLOADSIZE, PAYLOADSIZE);
            pkt.payload[PAYLOADSIZE] = '\0';
            calculate_checksum(&pkt);

            char packet[PACKETSIZE + 1];
            snprintf(packet, PACKETSIZE + 1, "%02d%s%s", pkt.seq_num, pkt.checksum, pkt.payload);

            int retval = send(sock, packet, PACKETSIZE, 0);
            if (retval == -1) {
                perror("send()");
                pthread_mutex_unlock(&mutex);
                break;
            }
            printf("packet %d is transmitted. (%s)\n", pkt.seq_num, pkt.payload);
            nextseqnum++;
        }
        pthread_mutex_unlock(&mutex);
        usleep(T * 1000000); // 0.5초 대기
    }
    return NULL;
}

void *recv_thread(void *arg) {
    while (1) {
        char ack[3];
        int retval = recv(sock, ack, 3, 0);
        if (retval <= 0) {
            perror("recv()");
            break;
        }
        ack[2] = '\0';
        int ack_num = atoi(ack);

        pthread_mutex_lock(&mutex);
        if (ack_num == expectedACK) {
            printf("(ACK = %d) is received.\n", ack_num);
            base = ack_num + 1;
            if (cwnd < ssthresh) {
                cwnd *= 2;
            } else {
                cwnd += 1;
            }
            duplicateACKs = 0;
        } else {
            duplicateACKs++;
            if (duplicateACKs == 3) {
                ssthresh = cwnd / 2;
                cwnd = 1;
                nextseqnum = base;
                printf("3 duplicate ACKs received, retransmitting from packet %d\n", base);
            }
        }
        expectedACK = ack_num + 1;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc > 1) SERVERIP = argv[1];

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        perror("socket()");
        return -1;
    }

    struct sockaddr_in serveraddr;
    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    inet_pton(AF_INET, SERVERIP, &serveraddr.sin_addr);
    serveraddr.sin_port = htons(SERVERPORT);
    if (connect(sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) == -1) {
        perror("connect()");
        close(sock);
        return -1;
    }

    fp = fopen("input1.txt", "r");
    if (!fp) {
        perror("fopen()");
        close(sock);
        return -1;
    }
    fread(txt, sizeof(char), 512, fp);
    fclose(fp);
    txt[512] = '\0';

    pthread_t send_id, recv_id;
    pthread_create(&send_id, NULL, send_thread, NULL);
    pthread_create(&recv_id, NULL, recv_thread, NULL);

    pthread_join(send_id, NULL);
    pthread_join(recv_id, NULL);

    close(sock);
    return 0;
}
