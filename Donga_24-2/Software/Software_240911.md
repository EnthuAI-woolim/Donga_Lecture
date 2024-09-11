# Software_240911

### 문제. fib(n) 함수를 구현하시오(피보나치 수열) - python style

1.

```python
fib(n) = (
	tmp(0) = 0
	tmp(1) = 1
	for i=1 to n
		tmp(n) = tmp(n-1) + tmp(n-2)
)
return tmp(n)
```

2.

```python
fib(o) = 1
fib(1) = 1
fib(n+2) = fib(n+1) + fib(n)
```

3.

```python
// 컴1
fib(0, 1), fib(1, 1)
fib(x+2) = fib(x+1) + fib(x)

// 컴2
write(x), fib(2, x) // 컴1

// 컴3
write(x), fib(3, x) // 컴1, 컴2

// 컴4
write(x), fib(4, x) // 컴1, 컴2, 컴3

// 컴5
write(x), fib(5, x) // 컴1, 컴3, 컴4

// 컴6
write(x), fib(5, x) // 컴5
```

### 문제. fib(n)을 계산하기 위해 주연 배우가 몇 명 필요한가?

=> n명