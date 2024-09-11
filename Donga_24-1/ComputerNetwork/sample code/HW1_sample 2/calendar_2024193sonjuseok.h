#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// 함수 정의
char* print_calendar(int year, int month);
int get_firstday(int year, int month);
int get_days(int year, int month);
int check_leapyear(int year);
char* calendar(char *buf);

// 캘린더 생성 후 변수에 저장하는 함수
char* print_calendar(int year, int month){
    char* buf = (char*)malloc(512 * sizeof(char));
    sprintf(buf, " SUN MON TUE WED THU FRI SAT\n\n");

    // 해당 일자의 캘린더 규격 설정
    int firstday = get_firstday(year, month);
    int days = get_days(year, month);
    for(int i=0; i<firstday; i++){
        strcat(buf, "    ");
    }
    for(int i=1; i<=days; i++){
        char day[5];
        sprintf(day, "%4d", i);
        strcat(buf, day);
        if((i+firstday)%7 == 0)
            strcat(buf, "\n");
    }
    strcat(buf, "\n");
    
    return buf;
}

// 입력받은 month의 시작 요일 계산
int get_firstday(int year, int month){
    int sum = 0;

    for(int i=1800; i<year; i++){
        if(check_leapyear(i))
            sum += 366;
        else
            sum += 365;
    }

    for (int i=1; i<month; i++){
        sum += get_days(year, i);
    }
        

    return (3 + sum) % 7; // 1800년 1월 1일이 수요일이기때문에 3을 더함
}

// 해당 년월의 일자 계산
int get_days(int year, int month){
    switch(month){
            case 1:
            case 3:
            case 5:
            case 7:
            case 8:
            case 10:
            case 12: 
                return 31;
            case 4:
            case 6:
            case 9:
            case 11:
                return 30;
            case 2:
                if(check_leapyear(year))
                    return 29;
                else
                    return 28;
            default:
                return -1;
        }
}

// 윤년 계산
int check_leapyear(int year){
    if((year%4 == 0) && (year%100 != 0) || (year%400 == 0))
        return 1;
    else
        return 0;
}

// TCPServer 호출 및 응답 함수
char* calendar(char *input){

    int year = 0;
    int month = 0;

    for(int i=0; i<4; i++){
        year = year * 10 + (input[i] - '0');
    }

    for(int i=6; i<8; i++){
        month = month * 10 + (input[i] - '0');
    }

    char* buf = print_calendar(year, month);

    return buf;
}