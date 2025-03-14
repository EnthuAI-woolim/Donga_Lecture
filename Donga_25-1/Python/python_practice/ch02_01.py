# _ : 파이썬 내부에서 사용하는 일부 예약어의 앞뒤에 사용
# ex) __main__

# num = int(input("숫자를 입력하세요\n-> "))
# print(num + 10)

msg = int('500')
# print(msg + num)

print('500' * 2)

# float_num = float(input("실수를 입력하세요\n-> "))

## f-string => f'' or f""
print(f'{msg} + 10 = {msg + 10}')

x = 3.14159
print("{:.2f}".format(x))  # 출력: 3.14
print(f"{x:.2f}")          # 출력: 3.14

## 튜플
my_tuple = (1, 2, 3)
my_tuple2 = 4, 5, 6     # 이렇게도 튜플 선언 가능

print(my_tuple)
print(my_tuple2)

