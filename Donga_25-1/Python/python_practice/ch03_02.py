# 하나의 정보체(변수)가 여러개의 값을 가지고 있을 때
# 값의 개수만큼 변수를 제공하면 값을 각각 담아준다(unpacking)

my_tuple = (1, 2, 3)
a, b, c = my_tuple
print(f'Unpacked values: a = {a}, b = {b}, c = {c}')

a, b, c = 4, 5, 6
print(f'Unpacked values: a = {a}, b = {b}, c = {c}')

d, e = 7, 8
d, e = e, d
print(f'Swapped values: d = {d}, e = {e}')

# num1, num2, num3 = 10, 20
# print(num1, num2, num3)
# num1, num2 = 100
# print(num1, num2)
num1 = 100, 200
print(num1)