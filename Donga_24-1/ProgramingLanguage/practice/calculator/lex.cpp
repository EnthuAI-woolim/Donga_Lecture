/* front.c-단순 산술식에 대한 어휘 분석기 시스템*/
#include<stdio.h>
#include<ctype.h>

/* 전역 변수 선언들*/
int charClass;
char lexeme[100];
char nextChar;
int lexLen;
int token;
int nextToken;
FILE *in_fp;

/* 함수 선언들*/
void addChar();
void getChar();
void getNonBlank();
int lex();

/* 문자 유형들 */
#define LETTER 0
#define DIGIT 1
#define UNKNOWN 99

/* 토큰 코드들 */
#define INT_LIT 10
#define IDENT 11
#define ASSIGN_OP 20
#define ADD_OP 21
#define SUB_OP 22
#define MULT_OP 23
#define DIV_OP 24
#define LEFT_PAREN 25
#define RIGHT_PAREN 26


#define LEX_MODE

/*********************************************************/
/* main 구동기*/
/* 입력 데이터 파일을 열고 그 내용을 처리*/
#ifdef LEX_MODE
int main() {
  if((in_fp = fopen("front.in", "r")) == NULL) 
    printf("ERROR-cannot open front.in \n");
  else {
    getChar();
    do {
      lex();
    } while (nextToken != EOF);
  }
}
#endif

/*********************************************************/
/* lookup –연산자와 괄호를 찾아서 토큰을 돌려주는 함수*/
int lookup (char ch) {
  switch (ch) {
    case '(':
      addChar();
      nextToken = LEFT_PAREN;
      break;
    case ')':
      addChar();
      nextToken = RIGHT_PAREN;
      break;
    case '+':
      addChar();
      nextToken = ADD_OP;
      break;
    case '-':
      addChar();
      nextToken = SUB_OP;
      break;
    case '*':
      addChar();
      nextToken = MULT_OP;
      break;
    case '/':
      addChar();
      nextToken = DIV_OP;
      break;
    default:
      addChar();
      nextToken = EOF;
      break;
  }
  return nextToken;
}

/*****************************************************/
/* addChar – 어휘항목에 nextChar를 추가하는 함수*/
void addChar() {
  if (lexLen <= 98) {
    lexeme[lexLen++] = nextChar;
    lexeme[lexLen] = 0;
  }
  else
    printf("Error-lexeme is too long\n");
}

/*****************************************************/
/* getChar - 입력으로부터 다음 번째 문자를 가져와서 그 문자 유형을 결정하는 함수 */
void getChar() {
  if ((nextChar = getc(in_fp)) != EOF) {
    if (isalpha(nextChar))
      charClass = LETTER;
    else if (isdigit(nextChar))
      charClass = DIGIT;
    else 
      charClass = UNKNOWN;
  }
  else
    charClass = EOF;
}

/*****************************************************/
/* getNonBlank – 공백을 건너뛰고 공백이 아닌 문자를 돌려줄때까지 getChar를 호출하는 함수 */
void getNonBlank() {
  while (isspace(nextChar))
    getChar();
}

/*****************************************************/
/* lex – 수식을 분석하는 단순 어휘 분석기 */
int lex() {
  lexLen = 0;
  getNonBlank();
  switch (charClass) {
    /* 식별자 파싱 */
    case LETTER:
      addChar();
      getChar();
      while (charClass == LETTER || charClass == DIGIT) {
        addChar();
        getChar();
      }
      nextToken = IDENT;
      break;
    
    /* 정수 리터럴 파싱 */
    case DIGIT:
      addChar();
      getChar();
      while (charClass == DIGIT) {
        addChar();
        getChar();
      }
      nextToken = INT_LIT;
      break;

    /* 괄호와 연산자 */
    case UNKNOWN:
      lookup(nextChar);
      getChar();
      break;

    /* EOF */
    case EOF:
      nextToken = EOF;
      lexeme[0] = 'E';
      lexeme[1] = 'O';
      lexeme[2] = 'F';
      lexeme[3] = '\0';
      break;
  } /* 스위치의 끝부분 */
  printf("Next token is: %d, Next lexeme is %s\n", nextToken, lexeme);
  return nextToken;
} /* 함수 lex의 끝부분 */
