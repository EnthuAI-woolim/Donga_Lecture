def coin_change(units, value):
    result = [0]*len(units)
    for i, unit in enumerate(units):
        while value >= unit:
            value -= unit
            result[i] += 1

    return result

if __name__ == "__main__":
    units = [500, 100, 50, 10]

    value = int(input("바꿀 액수 입력: ").strip())    
    if value == 0 or value < 0: 
        print("The total value cannot be zero or negative.")
    else:
        print(f"- {value} Won -")
        result = coin_change(units, value)
        # Print result
        for i in range(len(units)):
            print(f"{units[i]:>3} won: {result[i]}")