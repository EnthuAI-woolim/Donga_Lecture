const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);

const JwtProvider = require('../jwt/provider/JwtProvider');


// 로그인 화면 불러오기, 레이아웃 미사용
router.get('/login', function(req, res, next) {
    res.render('auth/login',  { layout: false, message: null });
});

// 회원가입 화면 불러오기, 레이아웃 미사용
router.get('/signup', function(req, res, next) {
    res.render('auth/signup', { layout: false, message: null });
});



// 회원가입 정보 저장하기
router.post('/signup', async function(req, res, next) {
    const { id, password, nickname } = req.body;

    try {
        const [ existingUsers ] = await query('SELECT * FROM member WHERE mem_id = ?', [id]);
        if (existingUsers) {
            return res.json({ success: false, message: '이미 존재하는 사용자입니다.' });
        }

        //              insert            //
        // description: 새로운 사용자 추가
        await query('INSERT INTO member (mem_id, mem_pw, nickname) VALUES (?, ?, ?)', [id, password, nickname]);
        // description: mainhome 초기화
        await query('INSERT INTO mainhome (mem_id, count_today, count_total) VALUES (?, 0 , 0)', [id]);
        await query('INSERT INTO miniroom (mem_id) VALUES (?)', [id]);

        //              render            //
        // description: 회원가입 성공 메시지
        res.json({ success: true, message : '회원가입 성공!' });
    } catch (err) {
        console.error('회원가입 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
        next(err);
    }
});

// 로그인 하기
router.post('/login', async function(req, res, next) {
    const { id, password } = req.body;
    try {
        const [user] = await query('SELECT * FROM member WHERE mem_id = ? AND mem_pw = ?', [id, password]);
        if (user) {
            const token = JwtProvider.generateToken(id);

            res.cookie('token', token, { httpOnly: true, secure: true, maxAge: 3600000 });
            res.json({ success: true, message: '로그인 성공!' });
        } else {
        res.json({ success: false, message: '존재하지 않는 회원입니다.' });
        }
    } catch (err) {
        console.error('로그인 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 로그아웃 하기
router.post('/logout', function(req, res, next) {
    try {
        res.clearCookie('token'); // 쿠키 삭제
        res.json({ success: true, message: '로그아웃 하셨습니다!' });
    } catch (err) {
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

module.exports = router;
