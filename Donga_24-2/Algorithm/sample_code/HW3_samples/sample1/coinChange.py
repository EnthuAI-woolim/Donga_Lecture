input = 2780 #동전으로 거슬러줘야 할 금액
fiveHundreds = 0 #500원 동전의 개수
oneHundreds = 0 #100원 동전의 개수
fifty = 0 #50원 동전의 개수
ten = 0 #10원 동전의 개수
while input > 0: #거슬러 줄 금액이 남아있는 동안 반복한다.
    if input // 500 != 0: #남아있는 금액이 500원 이상이면 500원 동전을 거슬러준다. (몫 만큼)
        fiveHundreds = input // 500
        input -= 500*(input//500)
    elif input // 100 != 0: #남아있는 금액이 100원 이상이면 100원 동전을 거슬러준다. (몫 만큼)
        oneHundreds = input // 100
        input -= 100*(input//100)
    elif input // 50 != 0: #남아있는 금액이 50원 이상이면 50원 동전을 거슬러준다. (몫 만큼)
        fifty = input // 50
        input -= 50*(input//50)
    elif input // 10 != 0: #남아있는 금액이 10원 이상이면 10원 동전을 거슬러준다. (몫 만큼)
        ten = input // 10
        input -= 10*(input//10)
result = f'{input} Won - 500 Won : {fiveHundreds}, 100 Won : {oneHundreds}, 50 Won : {fifty}, 10 Won : {ten}' #각 동전들의 개수를 출력해준다.
print(result)