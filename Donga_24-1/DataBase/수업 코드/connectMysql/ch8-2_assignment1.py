import pymysql
from tkinter import *
from tkinter import messagebox

def insertData() :
    con, cur = None, None
    data1, data2, data3, data4, data5, data6, data7, data8 = "", "", "", "", "", "", "", ""
    sql = ""

    conn = pymysql.connect(host='127.0.0.1', user='root', password='root', db='market_db', charset='utf8')
    cur = conn.cursor()

    data1 = edt1.get(); data2 = edt2.get(); data3 = edt3.get(); data4 = edt4.get()
    data5 = edt5.get(); data6 = edt6.get(); data7 = edt7.get(); data8 = edt8.get()
    sql = "INSERT INTO member VALUES('" + data1 + "','" + data2 + "'," + data3 + ",'" + data4 + "','" + data5 + "','" + data6 + "'," + data7 + ",'" + data8 + "')"
    cur.execute(sql)

    conn.commit()
    conn.close()
    messagebox.showinfo('성공', '데이터 입력 성공')

def selectData() :
    strData1, strData2, strData3, strData4, strData5, strData6, strData7, strData8 = [], [], [], [], [], [], [], []
    conn = pymysql.connect(host='127.0.0.1', user='root', password='root', db='market_db', charset='utf8')
    cur = conn.cursor()
    cur.execute("SELECT * FROM member")

    strData1.append("mem_id"); strData2.append("mem_name")
    strData3.append("mem_number"); strData4.append("addr")
    strData5.append("phone1"); strData6.append("phone2")
    strData7.append("height"); strData8.append("debut_date")
    strData1.append("-----------"); strData2.append("-----------"); strData3.append("-----------"); strData4.append("-----------")
    strData5.append("-----------"); strData6.append("-----------"); strData7.append("-----------"); strData8.append("-----------")

    while (True) :
        row = cur.fetchone()
        if row== None :
            break;
        strData1.append(row[0]); strData2.append(row[1]); strData3.append(row[2]); strData4.append(row[3])
        strData5.append(row[4]); strData6.append(row[5]); strData7.append(row[6]); strData8.append(row[7])

    # 리스트박스 내용 다 지우기
    listData1.delete(0,listData1.size() - 1); listData2.delete(0,listData2.size() - 1)
    listData3.delete(0,listData3.size() - 1); listData4.delete(0,listData4.size() - 1)
    listData5.delete(0, listData5.size() - 1); listData6.delete(0, listData6.size() - 1)
    listData7.delete(0, listData7.size() - 1); listData8.delete(0, listData8.size() - 1)

    for item1, item2, item3, item4, item5, item6, item7, item8 in zip(strData1, strData2, strData3, strData4, strData5, strData6, strData7, strData8):
        listData1.insert(END, item1); listData2.insert(END, item2)
        listData3.insert(END, item3); listData4.insert(END, item4)
        listData5.insert(END, item5); listData6.insert(END, item6)
        listData7.insert(END, item7); listData8.insert(END, item8)

    conn.close()

## 메인 코드부
root = Tk()
root.geometry("1200x600")
root.title("완전한 GUI 응용 프로그램")

edtFrame = Frame(root); edtFrame.pack()
listFrame = Frame(root); listFrame.pack(side = BOTTOM,fill=BOTH, expand=1)

edt1= Entry(edtFrame, width=10); edt1.pack(side=LEFT,padx=10,pady=10)
edt2= Entry(edtFrame, width=10); edt2.pack(side=LEFT,padx=10,pady=10)
edt3= Entry(edtFrame, width=10); edt3.pack(side=LEFT,padx=10,pady=10)
edt4= Entry(edtFrame, width=10); edt4.pack(side=LEFT,padx=10,pady=10)
edt5= Entry(edtFrame, width=10); edt5.pack(side=LEFT,padx=10,pady=10)
edt6= Entry(edtFrame, width=10); edt6.pack(side=LEFT,padx=10,pady=10)
edt7= Entry(edtFrame, width=10); edt7.pack(side=LEFT,padx=10,pady=10)
edt8= Entry(edtFrame, width=10); edt8.pack(side=LEFT,padx=10,pady=10)


btnInsert = Button(edtFrame, text="입력", command=insertData)
btnInsert.pack(side=LEFT,padx=10,pady=10)
btnSelect = Button(edtFrame, text="조회", command=selectData)
btnSelect.pack(side=LEFT,padx=10,pady=10)

listData1 = Listbox(listFrame,bg = 'yellow'); listData1.pack(side=LEFT,fill=BOTH, expand=1)
listData2 = Listbox(listFrame,bg = 'yellow'); listData2.pack(side=LEFT,fill=BOTH, expand=1)
listData3 = Listbox(listFrame,bg = 'yellow'); listData3.pack(side=LEFT,fill=BOTH, expand=1)
listData4 = Listbox(listFrame,bg = 'yellow'); listData4.pack(side=LEFT,fill=BOTH, expand=1)
listData5 = Listbox(listFrame,bg = 'yellow'); listData5.pack(side=LEFT,fill=BOTH, expand=1)
listData6 = Listbox(listFrame,bg = 'yellow'); listData6.pack(side=LEFT,fill=BOTH, expand=1)
listData7 = Listbox(listFrame,bg = 'yellow'); listData7.pack(side=LEFT,fill=BOTH, expand=1)
listData8 = Listbox(listFrame,bg = 'yellow'); listData8.pack(side=LEFT,fill=BOTH, expand=1)

root.mainloop()
