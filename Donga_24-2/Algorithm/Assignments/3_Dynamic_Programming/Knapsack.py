C = 10
items = [(5, 10), (4, 40), (6, 30), (3, 50)] # (무게, 가치)
n = len(items)
K = [[0 for _ in range(C+1)] for _ in range(n+1)]

for i in range(n+1): K[i][0] = 0
for w in range(C+1): K[0][w] = 0

for i in range(1, n+1):
    for w in range(1, C+1):
        if (items[i-1][0] > w): 
            K[i][w] = K[i-1][w]
        else: 
            K[i][w] = max(K[i-1][w], K[i-1][w - items[i-1][0]] + items[i-1][1])


for i in range(C+1): 
    if i == 0: 
        print(f"{'배낭 용량 w':^11}", end=" ") 
    print(f"{i:2d}", end=" ")
print()

for i in range(n+1):
    if i == 0: 
        print(f"{'물건':^3}{'가치':^3}{'무게':^3}", end=" ")
    
    for j in range(C+1):
        if j == 0 and i != 0:  print(f"{i:^5}{items[i-1][1]:^5}{items[i-1][0]:^5}", end=" ")
        print(f"{K[i][j]:2d}", end=" ")
    print()

