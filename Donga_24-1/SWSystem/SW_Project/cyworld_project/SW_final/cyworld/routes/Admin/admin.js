const express = require('express');
const router = express.Router();
const db = require('../../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);

const MemberListDto = require('../../dto/MemberListDto');
const ReportListDto = require('../../dto/ReportListDto');


// 관리자 페이지 불러오기
router.get('/home', async function(req, res, next) {
    try {
        const [ result ] = await query('SELECT count(*) AS count FROM member');
        const totalMemberNum = result.count;

        const memberList = await query('SELECT mem_id, mem_pw, nickname, created_at FROM member');
        const memberListDto = memberList.length > 0 ? memberList.map(item => new MemberListDto(item)) : [];

        const reportList = await query("SELECT reporter_id, reported_id, reason, report_time  FROM report");
        const reportListDto = reportList.length > 0 ? reportList.map(item => new ReportListDto(item)) : [];

        res.render('Admin/adminHome', { success: true, message: null,
            totalMemberNum: totalMemberNum,
            memberListDto: memberListDto,
            reportListDto: reportListDto
         });
    } catch (err) {
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 로그인 화면 불러오기, 레이아웃 미사용
router.get('/login', function(req, res, next) {
    res.render('Admin/adminLogin',  { message: null });
});

// 로그인 하기
router.post('/login', async function(req, res, next) {
    const { id, password } = req.body;
    try {
        const [ verifyAdmin ] = await query('SELECT is_admin FROM member WHERE mem_id = ? AND mem_pw = ?', [id, password]);
        if (verifyAdmin && verifyAdmin.is_admin == 1) {
            
            res.json({ success: true, message: '로그인 성공!' });
        } else {
            res.json({ success: false, message: '로그인 실패!' });
        }
    } catch (err) {
        console.error('로그인 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

module.exports = router;