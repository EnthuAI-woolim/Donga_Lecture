var {createServer} = require('http');
var fs = require('fs');
var url = require('url');
var qs = require('querystring');

var app = createServer((req, res)=> {
    var _url = req.url;
    var path = url.parse(_url, true).pathname;
    if(path === '/'){        
        fs.readFile(`./index.html`, (err,data)=>{                
            res.writeHead(200, {"Content-type":"text/html"});
            res.end(data);
        });
    }else if(path === '/post'){
        fs.readFile(`./post.html`, (err,data)=>{                
            res.writeHead(200, {"Content-type":"text/html"});
            res.end(data);
        });
    }else if(path === '/postProcess'){
        var body = '';
        req.on('data', function(data){
            body = body + data;
            console.log("body : " + body);
        });
        req.on('end', function(){            
            var post = qs.parse(body);
            var name = post.name;
            var age = post.age;
         
            res.writeHead(200, {"Content-type":"text/html"});
            res.end(`I am ${name}, ${age} years old.`);

        });    
    }else{
        res.writeHead(404);
        res.end("Not Found");
    }
});

app.listen(3000, ()=>{
    console.log("Server running...");
});
