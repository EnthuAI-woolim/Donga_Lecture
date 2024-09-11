function main(){
    const readline = require('readline').createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    readline.question('숫자를 입력해주세요 (1 또는 2): ', input => {
        if (input === '1') {
            console.log('hello');
        } else if (input === '2') {
            console.log('world');
        } else {
            console.log('1 또는 2를 입력해주세요.');
        }
    
        readline.close();
    });
}

main();