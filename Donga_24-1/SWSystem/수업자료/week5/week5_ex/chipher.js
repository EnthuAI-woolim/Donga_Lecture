const fs = require('fs');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function caesarCipher(text, shift) {
  return text.split('').map(char => {
    if (char.match(/[a-z]/i)) { // 정규표현식으로, i 플래그를 포함하여 a-z까지 문자를 대소문자 구분없이 찾음
      let code = char.charCodeAt();
      
      if (code >= 65 && code <= 90) {
        return String.fromCharCode(((code - 65 + shift) % 26 + 26) % 26 + 65);
      } else if (code >= 97 && code <= 122) {
        return String.fromCharCode(((code - 97 + shift) % 26 + 26) % 26 + 97);
      }
    }
    return char;
  }).join('');
}

function ask() {
  rl.question('파일 경로를 입력해주세요 : ', (filePath) => {
    rl.question('암호화는 +, 복호화는 -를 입력해주세요(+,-) : ', (method) => {
      rl.question('암호화 키를 입력해주세요(정수) : ', (key) => {
        const shift = parseInt(key);        
        if (isNaN(shift)) {
            console.log('잘못된 키 값입니다. 숫자를 입력해주세요.');
            rl.close();
            return;
        }
        processFile(filePath, method, shift);        
      });      
    });
  });  
}

function processFile(filePath, method, shift) {
  fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
      console.error('파일을 읽는 동안 오류가 발생했습니다:', err);
      return;
    }

    const k = method === '+' ? shift : -shift;
    const proc = method === '+' ? 'enc' : 'dec';
    
    const processedText = caesarCipher(data, k);
    console.log('처리된 텍스트:', processedText);

    // 결과를 새 파일에 저장
    const outputPath = `${filePath}_${proc}`;
    fs.writeFile(outputPath, processedText, (err) => {
      if (err) {
        console.error('파일을 쓰는 동안 오류가 발생했습니다:', err);
      } else {
        console.log(`파일이 성공적으로 쓰여졌습니다: ${outputPath}`);
      }
    });
  });
  rl.close();
}

// 호출 흐름 : ask -> processFile -> caesarCipher
ask();
