#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

// 문자열 대신 enum
enum thread_statement { idle, want_in, in_cs }; 

int flag[4];
int turn;
int n = 4;

void critical_section(int i) {
	flag[i] = want_in;
	int j;

	while (1) {
		// 바쁜대기
		while (turn != i) { // turn 이 i가 아니면
			if (flag[turn] == idle) { // 다른 쓰레드의 턴인데, 해당 쓰레드의 flag[turn]가 idle 상태이면.. 
				turn = i; // 내 턴이 될 수 있다.
			}
		}

		// 다른 애들이 in_cs (critical Section에) 있는지 확인 (위에서 내 턴이 될 '수' 있다 이기 때문에 한 번 더 확인 필요
		j = 0;
		while ((j < n) && (j == i || flag[j] != in_cs)) {
			j++;
		}

		if (j >= n) { // 아무도 없다. 들어갈 수 있음.
			break;
		}
	}
	flag[i] = in_cs;

	// 숫자 출력 
	for (int v = 25 * i + 1; v <= 25 * i + 25; v++) {
		printf("%d ", v * 3);
	}
	printf("\n");
}


void *thread_function(void *arg) {
	int thread_id = *((int*)arg);
	
	// 시작
	printf("<Thread %d STRT>\n", thread_id);

	critical_section(thread_id); // 0 ~ 3

	printf("<Thread %d END>\n", thread_id);

	// 종료
	flag[thread_id] = idle;
}

int main() {
	int args[4]; // thread_create의 *args
	pthread_t threads[4]; 
	turn = 0; // 아무거나 초기화해도 상관 X 

	// 시작 전 기본은 idle임 모두 초기화
	for (int i = 0; i < 4; i++) {
		flag[i] = idle;
	}

	// thread 생성
	for (int j = 0; j < 4; j++) {
		args[j] = j;
		pthread_create(&threads[j], NULL, thread_function, &args[j]);
	} 

	// Thread Join으로 리소스 돌려받기
	for (int k = 0; k < 4; k++) {
		pthread_join(threads[k], NULL);
	}
	
	return 0;
}

