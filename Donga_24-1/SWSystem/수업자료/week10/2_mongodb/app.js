// npm install express mongodb
const express = require('express');
const {MongoClient} = require('mongodb');

// Express 설정
const app = express();
const port = 3000;

// MongoDB 데이터베이스 연결 설정
const url = "mongodb://localhost:27017";
const client = new MongoClient(url, { useNewUrlParser: true, useUnifiedTopology: true });

async function connectToMongoDB() {
  try {
    await client.connect();
    console.log('Connected to MongoDB');
    return client.db('users'); // 데이터베이스 이름 설정
  } catch (err) {
    console.error(err);
    process.exit(1); // 연결 실패 시 서버 종료
  }
}

// MongoDB 데이터베이스 연결
const db = connectToMongoDB();

// JSON 파싱 미들웨어
app.use(express.json());

// 모든 사용자 정보 조회
app.get('/users', async (req, res) => {
  try {
    const database = await db;
    const users = database.collection('user');
    const query = {}; // 쿼리가 없다면 모든 정보를 가져옴 (select * from)
    const results = await users.find(query).toArray();
    res.status(200).json(results);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Database query error" });
  }
});

// 특정 사용자 정보 조회, :id는 변수처럼 사용한다는 의미
// 해당 위치에 id 값인 1, 2, 3등을 넣으면 됨 /users/1, /users/2 이런 식
app.get('/users/:id', async (req, res) => {
  const database = await db;
  const users = database.collection('user');
  const userId = parseInt(req.params.id); // URL로부터 받은 id를 정수로 변환

  try {
      // 숫자 id를 사용하여 문서를 조회
      const result = await users.findOne({ id: userId });
      if (result) {
          res.status(200).json(result);
      } else {
          res.status(404).send('User not found');
      }
  } catch (err) {
      console.error(err);
      res.status(500).json({ error: "Database query error" });
  }
});


// 사용자 생성
app.post('/users', async (req, res) => {
  const database = await db;
  const users = database.collection('user');

  try {
      const result = await users.insertOne(req.body);
      res.status(201).json(result);
  } catch (err) {
      console.error(err);
      res.status(500).json({ error: "Error inserting data" });
  }
});

// 사용자 정보 수정
app.put('/users/:id', async (req, res) => {
  const database = await db;
  const users = database.collection('user');
  const userId = parseInt(req.params.id);

  try {
      const result = await users.updateOne(
          { id: userId },
          { $set: req.body }
      );
      if (result.modifiedCount === 1) {
          res.status(200).json(result);
      } else {
          res.status(404).send('User not found or no changes made');
      }
  } catch (err) {
      console.error(err);
      res.status(500).json({ error: "Error updating data" });
  }
});

// 사용자 삭제
app.delete('/users/:id', async (req, res) => {
  const database = await db;
  const users = database.collection('user');
  const userId = parseInt(req.params.id);

  try {
      const result = await users.deleteOne({ id: userId });
      if (result.deletedCount === 1) {
          res.status(204).send();
      } else {
          res.status(404).send('User not found');
      }
  } catch (err) {
      console.error(err);
      res.status(500).json({ error: "Error deleting data" });
  }
});

// 서버 시작
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
