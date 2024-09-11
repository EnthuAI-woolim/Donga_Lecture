#include <stdio.h>
#include <string.h> // strtok 함수를 사용하기 위해 추가
#include <stdlib.h> // atoi 함수를 사용하기 위해 추가

#define FRAMES 3
#define MAX_PAGES 100

int main()
{
    int inputStream[MAX_PAGES]; // 최대 100개의 페이지를 입력 받을 수 있도록 설정
    int pageFaults = 0;
    int frames = 3;
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
    for (m = 0; m < FRAMES; m++)
    {
        temp[m] = -1;
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
                break;
            }
        }
        pageFaults++;
        if ((pageFaults <= FRAMES) && (s == 0))
        {
            temp[pageFaults - 1] = inputStream[m];
        }
        else if (s == 0)
        {
            // OPT 알고리즘: 앞으로 가장 오랫동안 사용되지 않을 페이지를 찾아 교체
            int farthest = m;
            int index = -1;
            for (n = 0; n < FRAMES; n++)
            {
                int j;
                for (j = m + 1; j < pages; j++)
                {
                    if (temp[n] == inputStream[j])
                    {
                        if (j > farthest)
                        {
                            farthest = j;
                            index = n;
                        }
                        break;
                    }
                }
                if (j == pages)
                {
                    index = n;
                    break;
                }
            }
            temp[index] = inputStream[m];
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
