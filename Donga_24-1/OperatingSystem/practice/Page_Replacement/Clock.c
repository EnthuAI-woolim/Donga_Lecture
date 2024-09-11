#include <stdio.h>
#include <string.h> // strtok 함수를 사용하기 위해 추가
#include <stdlib.h> // atoi 함수를 사용하기 위해 추가

#define FRAMES 3
#define MAX_PAGES 100

int main()
{
    int inputStream[MAX_PAGES]; // 최대 100개의 페이지를 입력 받을 수 있도록 설정
    int pageFaults = 0;
    int frames = FRAMES;
    int m, n, pages;

    // 페이지 스트림 입력 받기
    printf("페이지 스트림을 입력하세요 (쉼표로 구분하여 입력): ");
    char input[1000]; // 입력을 저장할 문자열 배열
    fgets(input, sizeof(input), stdin); // 한 줄을 입력 받음

    // 입력된 페이지 스트림 파싱하여 배열에 저장
    char *token = strtok(input, ",");
    int i = 0;
    while (token != NULL)
    {
        inputStream[i++] = atoi(token); // 문자열을 정수로 변환하여 배열에 저장
        token = strtok(NULL, ",");
    }
    pages = i;

    printf("입력 \t\t Frame 1 \t Frame 2 \t Frame 3\n");

    int temp[FRAMES];
    int refBit[FRAMES]; // 참조 비트
    int pointer = 0; // 시계 포인터

    for (m = 0; m < FRAMES; m++)
    {
        temp[m] = -1;
        refBit[m] = 0; // 초기화 시 참조 비트를 0으로 설정
    }

    // 페이지 부재가 발생하는 인덱스를 저장할 배열
    int pageFaultIndices[MAX_PAGES];
    int pageFaultIndexCount = 0;

    for (m = 0; m < pages; m++)
    {
        int found = 0;
        for (n = 0; n < FRAMES; n++)
        {
            if (inputStream[m] == temp[n])
            {
                found = 1;
                refBit[n] = 1; // 참조 비트를 설정
                break;
            }
        }

        if (found == 0)
        {
            pageFaults++;

            while (refBit[pointer] == 1)
            {
                refBit[pointer] = 0; // 참조 비트를 재설정
                pointer = (pointer + 1) % FRAMES; // 시계 방향으로 이동
            }

            // 교체할 페이지를 찾음
            temp[pointer] = inputStream[m];
            refBit[pointer] = 1; // 새 페이지의 참조 비트를 설정
            pointer = (pointer + 1) % FRAMES;

            // 페이지 부재가 발생하는 인덱스 저장
            pageFaultIndices[pageFaultIndexCount] = m;
            pageFaultIndexCount++;
        }

        printf("%d\t\t", inputStream[m]);
        for (n = 0; n < FRAMES; n++)
        {
            if (temp[n] != -1)
                printf(" %d\t\t", temp[n]);
            else
                printf(" 0 \t\t");
        }
        printf("\n");
    }

    printf("\n페이지 부재가 발생한 인덱스: ");
    for (int i = 0; i < pageFaultIndexCount; i++)
    {
        printf("%d ", pageFaultIndices[i]);
    }
    printf("\n");

    return 0;
}
