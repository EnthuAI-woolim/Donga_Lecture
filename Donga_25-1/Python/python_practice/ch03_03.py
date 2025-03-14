# n1 = 200
# n1 += 100
# print(n1)

total = 0
for _ in range(5):
    menu = input("구매 또는 판매한 메뉴 : ")
    price = int(input('가격(판매 +, 구매 -) : '))
    count = int(input('갯수 : '))
    total += price * count

    print(f'{menu}를 {price}에 판매/구매 했습니다.')
    print(f'현재 매출은 {total}원 입니다.')