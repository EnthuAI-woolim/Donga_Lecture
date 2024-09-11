const mysql = require('mysql');

const db = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: 'root',
  database: 'cyworld'
});

// MySQL 연결
db.connect((err) => {
  if (err) {
    console.error('Error connecting to MySQL database: ', err);
    return;
  }
  console.log('Connected to MySQL database');
});

module.exports = db;
