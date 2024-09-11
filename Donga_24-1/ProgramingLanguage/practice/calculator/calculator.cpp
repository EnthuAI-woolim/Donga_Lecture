#include<stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <cstring>

#define LETTER 0
#define DIGIT 1
#define UNKNOWN 99

#define INT_LIT 10
#define IDENT 11
#define ASSIGN_OP 20
#define ADD_OP 21
#define SUB_OP 22
#define MULT_OP 23
#define DIV_OP 24
#define LEFT_PAREN 25
#define RIGHT_PAREN 26
#define MAX_SYMBOLS 100

int expr();
int term();
int factor();
void lex();
void error();
void addChar();
void getChar();
void getNonBlank();
int lookupIdentifier();
int lookup();

int charClass; // 현재 문자의 유형
char lexeme[100]; // 현재 토큰
char nextChar; // 다음 문자
int lexLen; // 토큰의 길이
int token; // 현재 토큰 코드
int nextToken; // 다음 토큰 코드
FILE *in_fp;
char* symbolTable[MAX_SYMBOLS]; // 심볼 테이블 배열
int symbolValues[MAX_SYMBOLS]; // 각 심볼에 대한 값 배열
int numSymbols = 0; // 현재 심볼 테이블에 있는 심볼의 수

void addChar() {
  // 어휘 항목(lexeme)에 nextChar를 추가하는 함수
  if (lexLen <= 98) {
    lexeme[lexLen++] = nextChar;
    lexeme[lexLen] = 0; // 문자열 끝에 NULL 문자 추가
  }
  else
    printf("Error-lexeme is too long\n"); // 토큰이 너무 김
}

int lookup (char ch) {
  // 연산자와 괄호를 찾아서 해당 토큰 코드를 반환하는 함수
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
      nextToken = UNKNOWN; // 알 수 없는 문자일 경우 UNKNOWN 반환
      break;
  }
  return nextToken;
}

int expr() {
  // <expr> → <term> {( + | -) <term>}
  int result = term();

  while(nextToken == ADD_OP || nextToken == SUB_OP) {
    int op = nextToken; // 현재 연산자 저장
    lex();
    int termResult = term(); // 다음 term을 계산하여 결과를 얻음

    // 현재 연산자에 따라 계산 수행
    if (op == ADD_OP)
      result += termResult;
    else if (op == SUB_OP)
      result -= termResult;
  }

  return result;
}

int term () {
  // <term> → <factor> {( * | / ) <factor>)}
  int result = factor();
  
  while (nextToken == MULT_OP || nextToken == DIV_OP) {
    int op = nextToken; // 현재 연산자 저장
    lex();
    int factorResult = factor(); // 다음 factor을 계산하여 결과를 얻음

    // 현재 연산자에 따라 계산 수행
    if (op == MULT_OP)
      result *= factorResult;
    else if (op == DIV_OP)
      result /= factorResult;
  }

  return result;
}

int factor () {
  // <factor> → id | int_constant | (<expr>)
  int result;

  if (nextToken == IDENT) {
    result = lookupIdentifier(); // 식별자의 값을 반환
    lex(); // 다음 토큰으로 이동
  } else if (nextToken == INT_LIT) {
    result = atoi(lexeme); // 정수 리터럴의 값을 반환
    lex(); // 다음 토큰으로 이동
  } else if (nextToken == LEFT_PAREN) {
    lex(); // 다음 토큰으로 이동
    result = expr(); // 괄호 안의 표현식을 계산하여 결과를 반환
    if (nextToken == RIGHT_PAREN) {
        lex(); // 다음 토큰으로 이동
    } else {
        error(); // 오른쪽 괄호가 없는 경우 에러 처리
    }
  } else {
    error(); // 그 외의 경우에는 에러 처리
  }

  return result;
}

void error() {
  printf("Syntax error\n"); // 구문 오류 메시지 출력
}

int lookupIdentifier() {
  // 심볼 테이블에서 현재 식별자를 찾아서 해당 값을 반환
  for (int i = 0; i < numSymbols; i++) {
    if (strcmp(lexeme, symbolTable[i]) == 0) {
        return symbolValues[i]; // 식별자를 찾으면 해당 값을 반환
    }
  }
  printf("Error: Undefined identifier '%s'\n", lexeme); // 식별자가 없는 경우 에러 메시지 출력
  exit(1); // 프로그램 종료
}

void getChar() {
  // 입력으로부터 다음 문자를 가져와서 그 문자 유형을 결정하는 함수
  if ((nextChar = getc(in_fp)) != EOF) {
    if (isalpha(nextChar))
      charClass = LETTER; // 알파벳인 경우 LETTER로 설정
    else if (isdigit(nextChar))
      charClass = DIGIT; // 숫자인 경우 DIGIT로 설정
    else 
      charClass = UNKNOWN; // 그 외의 경우 UNKNOWN으로 설정
  }
  else
    charClass = EOF; // EOF인 경우 EOF로 설정
}

void getNonBlank() {
  // 공백을 건너뛰고 공백이 아닌 문자를 가져올 때까지 getChar를 호출하는 함수
  while (isspace(nextChar))
    getChar();
}

void lex() {
  // 수식을 분석하는 단순 어휘 분석기 함수
  lexLen = 0;
  getNonBlank();
  switch (charClass) {
    case LETTER:
      addChar();
      getChar();
      while (charClass == LETTER || charClass == DIGIT) {
        addChar();
        getChar();
      }
      nextToken = IDENT; // 식별자인 경우 IDENT 설정
      break;
    
    case DIGIT:
      addChar();
      getChar();
      while (charClass == DIGIT) {
        addChar();
        getChar();
      }
      nextToken = INT_LIT; // 정수 리터럴인 경우 INT_LIT 설정
      break;

    case UNKNOWN:
      lookup(nextChar); // 알 수 없는 문자 처리 추가
      getChar();
      break;

    case EOF:
      nextToken = EOF; // EOF인 경우 EOF 설정
      lexeme[0] = 'E';
      lexeme[1] = 'O';
      lexeme[2] = 'F';
      lexeme[3] = '\0'; // NULL 문자 추가
      break;
  }
  printf("Next token is: %d, Next lexeme is %s\n", nextToken, lexeme); // 디버깅용 출력문 추가
  
  // 알 수 없는 토큰이 발견되면 에러 처리
  if (nextToken == UNKNOWN) {
    error();
  }
}

int main() {
  // 입력 파일을 열고 그 내용을 처리하는 함수
  if((in_fp = fopen("front.in","r")) == NULL)
    printf("ERROR-cannot open front.in \n");
  else {
    getChar();
    do {
      lex(); // 어휘 분석기 함수 호출
      int result = expr(); // 표현식을 계산하는 함수 호출
      printf("Result: %d\n", result); // 결과 출력
    } while(nextToken != EOF);
  }

  return 0;
}