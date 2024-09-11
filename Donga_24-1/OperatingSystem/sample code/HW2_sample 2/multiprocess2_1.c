#include <stdio.h>
#include <time.h>

int main(){
    int var=3;
    clock_t start_time = clock();
    while(var<=9){
        for(int i=1; i<=100; i++){
            printf("%d ", i*var);
        }
        printf("\n");
        var+=2;
    }
    clock_t end_time = clock();
    printf("Single_process time: %lf\n", ((double)(end_time - start_time)) / CLOCKS_PER_SEC);

}