#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>

int main() {
  clock_t start_time = clock();

  for (int i = 1; i <= 100; i++)
    printf("%4d", i * 3);
  printf("\n");
  for (int i = 1; i <= 100; i++)
    printf("%4d", i * 5);
  printf("\n");
  for (int i = 1; i <= 100; i++)
    printf("%4d", i * 7);
  printf("\n");
  for (int i = 1; i <= 100; i++)
    printf("%4d", i * 9);
  printf("\n");

  clock_t end_time = clock(); 
  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("프로세스 종료까지 소요된 시간: %.7f 초\n", elapsed_time);
    
  return 0;
}