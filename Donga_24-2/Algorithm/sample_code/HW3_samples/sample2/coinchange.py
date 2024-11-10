def coinchange(money): 
        # 2780원 
        input_money = money
        ## 500원
        try_500 = money // 500
        for i in range(try_500):
            money -= 500
        ## left 280 
        try_100 = money // 100
        for i in range(try_100):
            money -= 100
        try_50 = money // 50
        for i in range(try_50):
            money -= 50
        try_10 = money // 10
        for i in range(try_10):
            money -= 10
        print("- ",input_money ," Won","- ","500 Won:", try_500, ", 100 Won:", try_100, ", 50 Won:", try_50, ", 10 Won:", try_10)
        return  try_500, try_100, try_50 , try_10
        
coinchange(2780)