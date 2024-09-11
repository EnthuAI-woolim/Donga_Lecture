import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', password='root', db='market_db', charset='utf8')
cur = conn.cursor()

# CREATE
# sql = "create table usertable (userid char(10), username char(10), email char(20), regyear int)"
# cur.execute(sql)
# conn.commit()
# conn.close()
# print('Good!')

# INSERT
# while (1) :
#     data1 = input("사용자 ID >> ")
#     if data1 == "q" or data1 == "Q":
#         break;
#
#     data2 = input("사용자 이름 >> ")
#     data3 = input("사용자 이메일 >> ")
#     data4 = input("가입 연도 >> ")
#
#     # (1)
#     sql = "INSERT INTO usertable VALUES('"+data1+"', '"+data2+"', '"+data3+"', "+data4+")"
#     cur.execute(sql)
#     # (2) cf) ch4-p.15
#     sql = "INSERT INTO usertable VALUES(%s, %s, %s, %s)"
#     data = (data1, data2, data3, data4)
#     cur.execute(sql, data)
#
# conn.commit()
# conn.close()
# print('Good!')

# SELECT
# cur.execute("select * from usertable")
# 
# print("사용자ID   사용자이름         이메일   가입연도")
# print("-----------------------------------------")
# while(True):
#     row = cur.fetchone()
#     if row == None: break
#
#     data1 = row[0]
#     data2 = row[1]
#     data3 = row[2]
#     data4 = row[3]
#     print("%5s %10s %15s %7d" %(data1, data2, data3, data4))
#
# conn.close()

sql = "DROP TABLE IF EXISTS pay2"
cur.execute(sql)

conn.commit()
conn.close()

