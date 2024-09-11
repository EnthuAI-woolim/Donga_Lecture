const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);
const layoutData = require('../public/middlewares/layout');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const MainhomeDto = require('../dto/MainhomeDto');
const layoutDto = require('../public/middlewares/layout');
const verifyingUser = require('../public/middlewares/verifyingUser');



// Multer 설정
const storage = multer.diskStorage({
    destination: function(req, file, cb) {
        cb(null, 'public/images/'); // 업로드할 디렉토리 설정
    },
    filename: function(req, file, cb) {
        const ext = path.extname(file.originalname); // 파일 확장자 추출
        cb(null, Date.now() + ext); // 파일 이름 설정 (현재 시간 + 확장자)
    }
});

const upload = multer({ storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 } // 파일 크기 제한 설정 (10MB)
 });
// 정적 파일 제공 (이미지 포함)
router.use('/images', express.static(path.join(__dirname, '../public/images')));



// /home/:id 라우트 처리
router.get('/home/:id', layoutData, MainhomeDto, (req, res) => {
    console.log('Rendering with mainhomeDto:', res.locals.mainhomeDto); // 데이터 확인
    if (!res.locals.mainhomeDto) {
        console.error('mainhomeDto is not defined');
        return res.status(500).send('mainhomeDto is not defined');
    }
    
    res.render('layout', {
        mainhomeDto: res.locals.mainhomeDto,
        miniroomDto: res.locals.miniroomDto,
        userId: res.locals.userId,
        myPage: res.locals.myPage,
        userListDto: res.locals.userListDto,
        updatednewsDto: res.locals.updatednewsDto
    });
});

// 프로필 이미지 변경하기
router.patch('/profile-edit/:id', verifyingUser, layoutDto, upload.single('profileImage'), async function(req, res, next) {
    const id =  req.params.id;
    if (!req.file) {
        return res.json({ success: false, id: id, message: '이미지 파일이 업로드되지 않았습니다.' });
    }

    const imageUrl = `/images/${req.file.filename}`; // 업로드된 이미지의 서버 상의 경로
    console.log(imageUrl);

    try {
        // description: 닉네임 변경
        await query('UPDATE member SET profile_picture = ? WHERE mem_id = ?', [imageUrl, id]);

        res.json({ success: true, id: id, message: '프로필 변경 성공!'});
    } catch (err) {
        console.error('닉네임 변경 중 오류 발생: ', err);
        res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 자기소개 글 변경하기
router.patch('/edit_intro/:id', async function(req, res, next) {
    const id = req.params.id;
    const { intro } = req.body;

    try {
        // description: 닉네임 변경
        await query('UPDATE mainhome SET profile_bio = ? WHERE mem_id = ?', [intro, id]);

        res.json({ success: true, id: id, message: '소개글 변경 성공!'});
    } catch (err) {
        console.error('소개글 변경 중 오류 발생: ', err);
        res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 닉네임 변경하기
router.patch('/edit_nickname/:id', async function(req, res, next) {
    const id = req.params.id;
    const { nickname } = req.body;

    try {
        // description: 닉네임 변경
        await query('UPDATE member SET nickname = ? WHERE mem_id = ?', [nickname, id]);

        res.json({ success: true, id: id, message: '닉네임 변경 성공!'});
    } catch (err) {
        console.error('닉네임 변경 중 오류 발생: ', err);
        res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 타이틀 변경하기
router.patch('/edit_title/:id', async function(req, res, next) {
    const id = req.params.id;
    const { title } = req.body;

    try {

        // description: 타이틀 변경
        await query('UPDATE mainhome SET title = ? WHERE mem_id = ?', [title, id]);

        res.json({ success: true, id: id, message: '타이틀 변경 성공!' });
    } catch (err) {
        console.error('타이틀 변경 중 오류 발생: ', err);
        res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

module.exports = router;