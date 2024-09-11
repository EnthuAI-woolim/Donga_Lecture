from tkinter import *
from tkinter import messagebox

# Button
def clickButton():
    messagebox.showinfo('버튼 클릭', '버튼을 눌렀습니다.')

root = Tk()
# =========================================
# 여기부터 코딩 추가

root.title("GUI 연습 화면")
root.geometry("400x200")    # 창 크기 설정

# 라벨 만들기
label1 = Label(root, text="SQL은")
label2 = Label(root, text="엄청 쉽습니다.", font=("휴먼편지체", 30), bg="green", fg="yellow")

label1.pack()   # pack()을 통해 해당 라벨을 화면에 표시
label2.pack()

# Button
button0 = Button(root, text="여기를 클릭하세요", fg="navy", bg="silver", command=clickButton)
button0.pack(expand=1)

# pack() 속성 : fill, padx, pady
button1 = Button(root, text="혼공1", command=clickButton)
button2 = Button(root, text="혼공2")
button3 = Button(root, text="혼공3")

button1.pack(side=LEFT, fill=X, padx=10, pady=10)
button2.pack(side=LEFT, fill=X, padx=10, pady=10)
button3.pack(side=LEFT, fill=X, padx=10, pady=10)


# =========================================
root.mainloop()