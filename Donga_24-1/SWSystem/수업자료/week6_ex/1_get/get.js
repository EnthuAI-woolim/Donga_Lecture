// get 실습 방법
// 브라우저에서 url에 아래와 같이 입력하여 접속
// localhost:3000/?name=jong&age=5

var http = require('http');
var url = require('url');
var app = http.createServer((req, res)=> {
    var _url = req.url;
    var query = url.parse(_url, true).query;
    if(req.method=='GET'){
        res.writeHead(200, {"Content-type":"text/html, charset=utf-8"});
          res.end(`I am ${query.name}, ${query.age} years old.`);
    }else{
        res.writeHead(404);
        res.end("Not Found");
    }
});
app.listen(3000);