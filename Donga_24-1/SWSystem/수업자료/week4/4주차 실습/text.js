const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function askText() {
  rl.question('영어로 텍스트를 입력해주세요: ', (text) => {
    askIndex(text);
  });
}

function askIndex(text) {
  rl.question(`인덱스를 입력해주세요 (0에서 ${text.length - 1} 사이): `, (index) => {
    if (index >= 0 && index < text.length) {
      askDirection(text, index);
    } else {
      console.log('유효하지 않은 인덱스입니다. 다시 입력해주세요.');
      askIndex(text);
    }
  });
}

function askDirection(text, index) {
  rl.question('인덱스부터 앞을 지울지 뒤를 지울지 선택해주세요. 앞 ("<"), 뒤 (">"): ', (direction) => {
    if (direction === '<') {
      console.log('결과:', text.substring(index));
      rl.close();
    } else if (direction === '>') {
      console.log('결과:', text.substring(0, parseInt(index) + 1));
      rl.close();
    } else {
      console.log('"<" 또는 ">"를 입력해주세요.');
      askDirection(text, index);
    }
  });
}

askText();
