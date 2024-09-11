const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);

const verifyingUser = require('../public/middlewares/verifyingUser');
const layoutDto = require('../public/middlewares/layout');
const postListDto = require('../dto/PostListDto');


// 다이어리 불러오기
router.get('/:id', verifyingUser, layoutDto, async function(req, res, next) {
    const id = req.params.id;
    try {
        const postListDto = await query('SELECT title, content, created_at, post_id FROM post WHERE mem_id = ? order by created_at DESC', [id]);

        res.render('post', { success: true, message: null, postListDto: postListDto });
    } catch (err) {
        res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
    
});

// 작성한 다이어리 저장하기
router.post('/write/:id', async function(req, res, next) {
    const id = req.params.id;
    const { title, content, date } = req.body;
    try {
        await query('INSERT INTO post (mem_id, title, content, post_date) VALUES (?, ?, ?, ?)', [id, title, content, date]);

        res.json({ success: true, message: '다이어리를 작성 성공!', title: title, date: date });
    } catch (err) {
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 다이어리 삭제
router.delete('/delete/:id', async function(req, res, next) {
    const id = req.params.id;
    console.log("Delete Post ID: " + id);
    try {
        // 다이어리 삭제 코드 작성
        await query('DELETE FROM post WHERE post_id = ?', [id]);
        res.status(200).json({ success: true, message: '다이어리가 성공적으로 삭제되었습니다.' });
    } catch (error) {
        console.error('다이어리 삭제 중 오류:', error);
        res.status(500).json({ success: false, message: '다이어리 삭제 중 오류가 발생했습니다.' });
    }
});


module.exports = router;
