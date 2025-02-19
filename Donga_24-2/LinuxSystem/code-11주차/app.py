from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
app.secret_key = "secret_key"  # 세션 관리를 위한 키

# MySQL 연결 설정 (Table은 직접 만들어야함)
# use your_db;             # 계정과 동일한 db를 만들어둚
# CREATE TABLE users (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     name VARCHAR(50) NOT NULL UNIQUE,
#     password VARCHAR(255) NOT NULL
# );
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="user",        # 본인 계정으로 수정
        password="user",    # mysql 비밀번호는 계정과 동일
        database="user"     # 계정과 동일한 db를 만들어둚
    )

@app.route("/", methods=["GET", "POST"])
def login():
    message = None
    if request.method == "POST":
        name = request.form.get("name")
        password = request.form.get("password")

        if not name or not password:
            message = "input all fields."
        else:
            try:
                connection = get_db_connection()
                cursor = connection.cursor(dictionary=True)
                query = "SELECT * FROM users WHERE name = %s AND password = %s"
                cursor.execute(query, (name, password))
                user = cursor.fetchone()
                if user:
                    session["user"] = name
                    return redirect(url_for("home"))
                else:
                    message = "wrong name or password."
            except Error as e:
                message = f"database error: {e}"
            finally:
                cursor.close()
                connection.close()

    return render_template("login.html", message=message)

@app.route("/register", methods=["GET", "POST"])
def register():
    message = None
    if request.method == "POST":
        name = request.form.get("name")
        password = request.form.get("password")

        if not name or not password:
            message = "input all fields."
        else:
            try:
                connection = get_db_connection()
                cursor = connection.cursor()
                query = "INSERT INTO users (name, password) VALUES (%s, %s)"
                cursor.execute(query, (name, password))
                connection.commit()
                return redirect(url_for("login"))
            except mysql.connector.IntegrityError:
                message = "already exists."
            except Error as e:
                message = f"database error: {e}"
            finally:
                cursor.close()
                connection.close()

    return render_template("register.html", message=message)

@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("home.html", user=session["user"])

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=15001)     #개인 포트 번호
