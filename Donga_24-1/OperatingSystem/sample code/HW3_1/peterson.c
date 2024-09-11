#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
bool flag[2] = { false, false };
int turn = 0; // 0 부터 시작

void * first_thread_fc(void *args) { // 0

	flag[0] = true; // 들어가고 싶다.
	turn = 1;		// 근데 1한테 턴을 양보할게
	while (flag[1] && turn == 1) { // 턴이 1이고 , 1쪽에서도 들어오고 싶은 의사가 있다면 
		// 대기(1이 일하게 두자 나는 가만히 있고)
		continue;
	}
	// 1이 끝나서 탈출 했다.
	
	// 임계구역
	printf("<Thread 1> STRT\n");
	for (int i = 1; i <= 51; i++) {
		printf("%d ", i * 3);
	}
	printf("\n<Thread 2> END\n");

	flag[0] = false; // 임계구역 끝 . 순서 넘기기
	
	return NULL;
} 

void *second_thread_fc(void *args) { // 1 
	flag[1] = true;
	turn = 0;

	while (flag[0] == true && turn == 0) {
		// 이미 0이 임계구역에 있으니 대기하기
		continue;
	}

	// 임계구역
	printf("<Thread 2> STRT\n");
	for (int i = 51; i <= 100; i++) {
		printf("%d ", i * 3);
	}
	printf("\n<Thread 2> END\n");

	flag[1] = false;	// 임계구역 끝 

	return NULL;
} 


int main()
{
	pthread_t th0, th1;

	pthread_create(&th0, NULL, first_thread_fc, NULL);
	pthread_create(&th1, NULL, second_thread_fc, NULL);

	pthread_join(th0, NULL);
	pthread_join(th1, NULL);

}
