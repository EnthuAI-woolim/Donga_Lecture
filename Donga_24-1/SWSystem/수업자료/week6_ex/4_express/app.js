// express 설치
// npm init -y
// npm install express

const express = require('express');
const app = express();
const path = require('path');
const port = 3000;

// POST를 처리하기 위한 부분
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// localhost:3000에 접속하면 index.html이 보임
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// get 실습 방법
// 브라우저에서 url에 아래와 같이 입력하여 접속
// localhost:3000/get/?name=jong&age=5
app.get('/get', (req, res) => {
    const name = req.query.name;
    const age = req.query.age;
  
    res.send(`get : I am ${name}, ${age} years old.`);
});

app.post('/post', (req, res) => {
    const name = req.body.name;
    const age = req.body.age;
  
    res.send(`<p>Post : I am ${name}, ${age} years old.</p>
    <button onclick="location.href='/'">Back</button>
    `);
  });

app.listen(port, () => {
  console.log(`Express server listening at http://localhost:${port}`);
});
