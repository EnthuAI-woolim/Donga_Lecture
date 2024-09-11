const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.question('주사위를 굴릴 횟수를 입력해주세요 (10~1000000): ', (input) => {
  const rolls = parseInt(input);
  if (rolls < 10 || rolls > 1000000 || isNaN(rolls)) {
    console.log('입력이 유효하지 않습니다. 프로그램을 종료합니다.');
    rl.close();
    return;
  }

  const results = rollDice(rolls);
  displayResults(results, rolls);
  rl.close();
});

function rollDice(rolls) {
  const results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0};
  for (let i = 0; i < rolls; i++) {
    const result = Math.floor(Math.random() * 6) + 1;
    results[result]++;
  }
  return results;
}

function displayResults(results, totalRolls) {
  console.log('주사위 굴리기 결과:');
  for (const [side, count] of Object.entries(results)) {
    const probability = ((count / totalRolls) * 100).toFixed(2); // 확률을 백분율로 변환
    console.log(`${side}이(가) 나온 횟수: ${count}, 확률: ${probability}%`);
  }
}
