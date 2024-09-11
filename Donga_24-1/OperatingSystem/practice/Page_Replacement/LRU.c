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
    int m, n, s, pages;

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

    printf("입력 \t\t Frame 1 \t Frame 2 \t Frame 3 ");

    int temp[FRAMES];
    int lastUsed[FRAMES]; // 각 프레임의 마지막 사용 시간을 저장하는 배열
    for (m = 0; m < FRAMES; m++)
    {
        temp[m] = -1;
        lastUsed[m] = -1; // 초기화
    }

    // 페이지 부재가 발생하는 인덱스를 저장할 배열
    int pageFaultIndices[MAX_PAGES];
    int pageFaultIndexCount = 0;

    for (m = 0; m < pages; m++)
    {
        s = 0;
        for (n = 0; n < FRAMES; n++)
        {
            if (inputStream[m] == temp[n])
            {
                s++;
                pageFaults--;
                lastUsed[n] = m; // 현재 시간을 마지막 사용 시간으로 업데이트
            }
        }
        pageFaults++;
        if ((pageFaults <= FRAMES) && (s == 0))
        {
            temp[pageFaults - 1] = inputStream[m];
            lastUsed[pageFaults - 1] = m; // 현재 시간을 마지막 사용 시간으로 설정
        }
        else if (s == 0)
        {
            // LRU 알고리즘: 가장 오랫동안 사용되지 않은 페이지를 찾아 교체
            int lru = 0;
            for (n = 1; n < FRAMES; n++)
            {
                if (lastUsed[n] < lastUsed[lru])
                {
                    lru = n;
                }
            }
            temp[lru] = inputStream[m];
            lastUsed[lru] = m; // 현재 시간을 마지막 사용 시간으로 업데이트
        }

        // 페이지 부재가 발생하는 인덱스 저장
        if (s == 0)
        {
            pageFaultIndices[pageFaultIndexCount] = m;
            pageFaultIndexCount++;
        }

        printf("\n");
        printf("%d\t\t", inputStream[m]);
        for (n = 0; n < FRAMES; n++)
        {
            if (temp[n] != -1)
                printf(" %d\t\t", temp[n]);
            else
                printf(" 0 \t\t");
        }
    }

    printf("\n페이지 부재가 발생한 인덱스: ");
    for (int i = 0; i < pageFaultIndexCount; i++)
    {
        printf("%d ", pageFaultIndices[i]);
    }
    printf("\n");

    return 0;
}
