const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);

const verifyingUser = require('../public/middlewares/verifyingUser');
const layoutDto = require('../public/middlewares/layout');;


router.get('/:id', verifyingUser, layoutDto, async function(req, res, next) {
    const id = req.params.id
    
    try {
        res.render('setting', { layout: true, message: null });
    } catch (err) {
        console.error('데이터를 불러오는 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
})

// 회원 신고
router.post('/member_report/:id', async function(req, res, next) {
    const id = req.params.id
    const userId = req.user.userId;
    const { reason } = req.body;
    
    try {
        await query('INSERT INTO report (reporter_id, reported_id, reason) VALUES (?, ?, ?)', [userId, id, reason]);
        res.json({ success: true, message: '회원 신고 하셨습니다!' });
    } catch (err) {
        console.error('데이터를 불러오는 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 회원 탈퇴
router.delete('/member_getout/:id', async function(req, res, next) {
    const id = req.params.id
    
    try {
        await query('DELETE FROM member WHERE mem_id = ?', [id]);
        res.clearCookie('token');
        res.json({ success: true, message: '회원 탈퇴 하셨습니다!' });
    } catch (err) {
        console.error('데이터를 불러오는 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});


module.exports = router;