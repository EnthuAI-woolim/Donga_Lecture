var http = require('http');
var fs = require('fs');
var url = require('url');

var app = http.createServer((req, res)=> {
    var _url = req.url;
    var path = url.parse(_url, true).pathname;

    if(path === '/'){        
        fs.readFile(`./index.html`, (err,data)=>{                
            res.writeHead(200, {"Content-type":"text/html"});
            res.end(data);
        });
    }else if(path === '/signup'){
        fs.readFile(`./signup.html`, (err,data)=>{                
            res.writeHead(200, {"Content-type":"text/html"});
            res.end(data);
        });
    }else{
        res.writeHead(404);
        res.end("Not Found");
    }
});

app.listen(3000, ()=> {
    console.log(`Server running at http://localhost:3000/`);
  });
  