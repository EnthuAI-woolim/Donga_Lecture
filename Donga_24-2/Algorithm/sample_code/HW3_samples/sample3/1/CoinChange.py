def CoinChange(W):
    n500 = n100 = n50 = n10 = 0
    change = W

    # 500원 동전 계산
    while change >= 500:
        change -= 500
        n500 += 1

    # 100원 동전 계산
    while change >= 100:
        change -= 100
        n100 += 1

    # 50원 동전 계산
    while change >= 50:
        change -= 50
        n50 += 1

    # 10원 동전 계산
    while change >= 10:
        change -= 10
        n10 += 1

    # 결과 출력
    print(f"{W} Won – 500 Won: {n500}, 100 Won: {n100}, 50 Won: {n50}, 10 Won: {n10}")

# 2780원의 거스름돈 계산
CoinChange(2780)
