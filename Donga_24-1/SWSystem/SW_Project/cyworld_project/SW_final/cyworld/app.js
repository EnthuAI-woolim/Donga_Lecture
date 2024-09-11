const createError = require('http-errors');
const express = require('express');
const path = require('path');
const cookieParser = require('cookie-parser');
const logger = require('morgan');

const expressLayouts = require('express-ejs-layouts'); // express-ejs-layouts 추가
const authenticateToken = require('./jwt/middleware/authMiddleware');

const layoutRouter = require('./routes/layout');
const authRouter = require('./routes/auth');
const adminRouter = require('./routes/Admin/admin');
const homeRouter = require('./routes/home');
const postRouter = require('./routes/post');
const photoRouter = require('./routes/photo');
const guestbookRouter = require('./routes/guestbook');
const searchRouter = require('./routes/search');
const friendRouter = require('./routes/friend');
const settingRouter = require('./routes/setting');

const app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/admin', adminRouter);

// express-ejs-layouts 사용
app.use(expressLayouts);
app.set('layout', 'layout'); 
app.set("layout extractScripts", true)


app.use('/auth', authRouter); 

app.use(authenticateToken);
app.use('/layout', layoutRouter);
app.use('/home', homeRouter); 
app.use('/post', postRouter); 
app.use('/photo', photoRouter); 
app.use('/guestbook', guestbookRouter); 
app.use('/friend', friendRouter);
app.use('/search', searchRouter);
app.use('/setting', settingRouter);



// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
