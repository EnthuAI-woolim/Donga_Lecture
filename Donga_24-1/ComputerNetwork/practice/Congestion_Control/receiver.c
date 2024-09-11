
#include <pthread.h>
#include "./Common.h"


#define SERVERPORT 9000
#define PACKETSIZE 14
#define PAYLOADSIZE 10
#define BUFSIZE 70
#define T 1

typedef struct {
    int seq_num;
    char payload[PAYLOADSIZE + 1];
    unsigned char checksum[2];
} Packet;

int server_sock;
char buffer[BUFSIZE + 1];
FILE *fp;
int base = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int verify_checksum(Packet *pkt) {
    int sum = 0;
    for (int i = 0; i < PAYLOADSIZE; i++) {
        sum += pkt->payload[i];
    }
    sum = (sum & 0xFF) + (sum >> 8);
    unsigned char checksum = ~(sum & 0xFF);
    return checksum == pkt->checksum[0];
}

void *recv_buffer_thread(void *arg) {
    while (1) {
        char packet[PACKETSIZE + 1];
        int retval = recv(server_sock, packet, PACKETSIZE, 0);
        if (retval <= 0) {
            perror("recv()");
            break;
        }
        packet[PACKETSIZE] = '\0';

        Packet pkt;
        sscanf(packet, "%02d%2s%10s", &pkt.seq_num, pkt.checksum, pkt.payload);
        pkt.payload[PAYLOADSIZE] = '\0';

        pthread_mutex_lock(&mutex);
        if (strlen(buffer) + PACKETSIZE <= BUFSIZE) {
            strcat(buffer, packet);
        } else {
            printf("packet is dropped!\n");
        }
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

void *process_buffer_thread(void *arg) {
    while (1) {
        usleep(T * 1000000); // 1초 대기

        pthread_mutex_lock(&mutex);
        if (strlen(buffer) >= PACKETSIZE) {
            Packet pkt;
            sscanf(buffer, "%02d%2s%10s", &pkt.seq_num, pkt.checksum, pkt.payload);
            pkt.payload[PAYLOADSIZE] = '\0';

            memmove(buffer, buffer + PACKETSIZE, strlen(buffer) - PACKETSIZE + 1);

            if (verify_checksum(&pkt)) {
                printf("packet %d is received and there is no error. (%s) ", pkt.seq_num, pkt.payload);
                fp = fopen("output1.txt", "a");
                if (fp) {
                    fputs(pkt.payload, fp);
                    fclose(fp);
                }
                char ack[3];
                snprintf(ack, 3, "%02d", pkt.seq_num + 1);
                send(server_sock, ack, 3, 0);
                printf("(ACK = %d) is transmitted.\n", pkt.seq_num + 1);
            } else {
                printf("packet %d is received and there is some error.\n", pkt.seq_num);
            }
        }
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main() {
    server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock == -1) {
        perror("socket()");
        return -1;
    }

    struct sockaddr_in serveraddr;
    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serveraddr.sin_port = htons(SERVERPORT);
    if (bind(server_sock, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) == -1) {
        perror("bind()");
        close(server_sock);
        return -1;
    }

    if (listen(server_sock, SOMAXCONN) == -1) {
        perror("listen()");
        close(server_sock);
        return -1;
    }

    printf("Server is listening on port %d\n", SERVERPORT);

    struct sockaddr_in clientaddr;
    socklen_t addrlen = sizeof(clientaddr);
    int client_sock = accept(server_sock, (struct sockaddr *)&clientaddr, &addrlen);
    if (client_sock == -1) {
        perror("accept()");
        close(server_sock);
        return -1;
    }
    printf("Client connected.\n");

    pthread_t recv_id, process_id;
    pthread_create(&recv_id, NULL, recv_buffer_thread, NULL);
    pthread_create(&process_id, NULL, process_buffer_thread, NULL);

    pthread_join(recv_id, NULL);
    pthread_join(process_id, NULL);

    close(client_sock);
    close(server_sock);
    return 0;
}
