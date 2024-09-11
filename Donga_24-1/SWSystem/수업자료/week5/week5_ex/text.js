const text = 'Hello World!';
console.log(text);
console.log(text.split(''));
console.log(text.split(' ').join(', '));


const text2 = 'abc';
console.log(text2);
console.log(text2.split(''));
console.log(text2.split('').map(t => {
    if(t === 'a'){
        t = 'e';
    }
    return t;
}));
console.log(text2.split('').map(t => {
    if(t === 'a'){
        t = 'e';
    }
    return t;
}).join(':')
);