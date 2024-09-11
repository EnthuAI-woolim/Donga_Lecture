const fs = require('fs');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function procFile() {
    rl.question('파일 내용을 입력해주세요 : ', (data) => {
        fs.writeFile('a.out', data, (err) => {
            if (err) {
              console.error('파일을 쓰는 동안 오류가 발생했습니다:', err);
            } else {
              console.log(`파일이 성공적으로 쓰여졌습니다.`);
            }
          });
        rl.close();
    });
}

procFile();