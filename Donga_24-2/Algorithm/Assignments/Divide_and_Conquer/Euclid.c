#include <stdio.h>

int main(int argc, const char *argv[]) {
    int a1 = 24;
    int a2 = 14;
    int r = 0;

    while(1) {
        r = a1 % a2;
        if (r == 0) {
            printf("%d\n", a2);
            return 0;
        }
        a1 = a2;
        a2 = r;
    }

    return 0;
}
