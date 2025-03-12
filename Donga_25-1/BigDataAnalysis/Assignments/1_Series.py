import pandas as pd


sales = pd.Series(
    [781, 650, 705, 406, 580, 450, 550, 640],
    index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
)

#%%
# 1.
# sales.where((sales < 500) | (sales > 700)).dropna()
sales.loc[(sales < 500) | (sales > 700)]

#%%
# 2.
# sales.where(sales > sales.loc['B']).dropna()
sales.loc[sales > sales.loc['B']]

#%%
# 3.
# sales.where(sales < 600).dropna().index
# under_600 = sales.loc[sales < 600].index.tolist()
under_600 = sales.loc[sales < 600].index
under_600

#%%
# 4. 
# sales.where(sales < 600).dropna() * 1.2
sales.loc[sales < 600] * 1.2

#%%
# 5. 
# sales.agg(['mean', 'sum', 'std'])
print('mean :', sales.mean())
print('sum :', sales.sum())
print('std :', sales.std())

#%%
# 6.
sales.loc[['A', 'C']] = [810, 820]
sales

#%%
# 7. 
# sales = sales._append(pd.Series({'J':400}))
sales.loc['J'] = 400
sales

#%%
# 8. 
# sales.drop('J', inplace=True)
sales = sales.drop('J')
sales

#%%
# 9.
sales2 = sales.copy()
sales2 = sales2 + 500
print(f'sales\n{sales}\n')
print(f'sales2\n{sales2}')

