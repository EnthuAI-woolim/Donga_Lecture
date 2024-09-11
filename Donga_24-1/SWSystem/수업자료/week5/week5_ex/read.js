const fs = require('fs');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function procFile() {
    rl.question('파일 경로를 입력해주세요 : ', (filePath) => {
        fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) {
              console.error('파일을 읽는 동안 오류가 발생했습니다:', err);
              return;
            }
            console.log('처리된 텍스트:', data);
        });
        rl.close();
    });
}

procFile();