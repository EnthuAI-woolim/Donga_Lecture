// npm install express
const express = require('express');
const app = express();
const port = 3000;

// npm install mysql
var mysql = require('mysql');
var db = mysql.createConnection({
  host:'localhost',
  user:'root',
  password:'1234',
  database:'user_db'
});
db.connect((err) => {
  if (err) {
    throw err;
  }
  console.log('Connected to database');
});

app.use(express.json()); // JSON 파싱 미들웨어

// 초기 사용자 데이터
// let users = [
//     { id: 1, name: 'Kim' },
//     { id: 2, name: 'Lee' },
//     { id: 3, name: 'Park' }];

// 모든 사용자 정보 조회
app.get('/users', (req, res) => {
  db.query('select * from users', (err, result) => {
    if (err){
        console.log(err);
    }
    console.log(result);
    res.status(200).json(result);
  });
});

// 특정 사용자 정보 조회, :id는 변수처럼 사용한다는 의미
// 해당 위치에 id 값인 1, 2, 3등을 넣으면 됨 /users/1, /users/2 이런 식
app.get('/users/:id', (req, res) => {
  db.query(`select * from users where id = ${req.params.id}`, (err, result) => {
    if (err){
        console.log(err);
    }
    console.log(result);
    res.status(200).json(result);
  });
});

// 사용자 생성
app.post('/users', (req, res) => {
  const user = {
    id: req.body.id,
    name: req.body.name
  };
  db.query(`insert into users(id, name) values(${user.id}, '${user.name}')`, (err, result) => {
    if (err){
        console.log(err);
    }
    console.log(result);
    res.status(200).json(result);
  });
});

// 사용자 정보 수정
// app.put('/users/:id', (req, res) => {
//   const user = users.find(u => u.id === parseInt(req.params.id));
//   if (user) {
//     user.name = req.body.name;
//     res.status(200).json(user);
//   } else {
//     res.status(404).send('User not found');
//   }
// });

// 사용자 삭제
// app.delete('/users/:id', (req, res) => {
//   users = users.filter(u => u.id !== parseInt(req.params.id));
//   res.status(204).send();
// });

// 서버 시작
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
