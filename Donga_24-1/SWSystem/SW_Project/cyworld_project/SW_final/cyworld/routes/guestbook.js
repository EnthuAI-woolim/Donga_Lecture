const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);

const verifyingUser = require('../public/middlewares/verifyingUser');
const layoutDto = require('../public/middlewares/layout');
const GuestbookListDto = require('../dto/GuestbookListDto');


// guestbook 불러오기
router.get('/:id', verifyingUser, layoutDto, async function(req, res, next) {
    const id = req.params.id;

    try {
        const guestbookListDto = await query(`
            SELECT A.guestbook_id, A.visitor_id, A.content, A.created_at, 
                   B.comment_id, B.mem_id, B.comment_text, B.created_at AS comment_created_at
            FROM guestbook A
            LEFT JOIN guestbook_comments B ON A.guestbook_id = B.guestbook_id
            WHERE A.mem_id = ?
            ORDER BY A.created_at DESC, B.created_at ASC    
        `, [id]);
        // ORDER BY A.created_at DESC, B.created_at ASC

        res.render('guestbook', { success: true, message: null, guestbookListDto: guestbookListDto });
    } catch (err) {
        console.error('방명록 불러오기 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 작성한 방명록 저장하기
router.post('/write/:id', async function(req, res, next) {
    const id = req.params.id;
    const userId = req.user.userId;
    const { content } = req.body;
    const created_at = new Date();

    try {
        await query('INSERT INTO guestbook (mem_id, visitor_id, content, created_at) VALUES (?, ?, ?, ?)', [id, userId, content, created_at]);
        res.json({ success: true, message: '방명록 작성이 성공적으로 완료되었습니다.' });
    } catch (err) {
        console.error('방명록 작성 중 오류:', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 방명록 삭제
router.delete('/delete/:id', async function(req, res, next) {
    const guestbookid = req.params.id;
    console.log("Delete Guestbook ID: " + guestbookid);
    try {
        // 방명록 삭제 코드 작성
        await query('DELETE FROM guestbook WHERE guestbook_id = ?', [guestbookid]);
        res.status(200).json({ success: true, message: '방명록이 성공적으로 삭제되었습니다.' });
    } catch (error) {
        console.error('방명록 삭제 중 오류:', error);
        res.status(500).json({ success: false, message: '방명록 삭제 중 오류가 발생했습니다.' });
    }
});

// 방명록 댓글 저장하기
router.post('/submit_comment', async function(req, res, next) {
    const userId = req.user.userId;
    const { guestbook_id, content } = req.body;
    const created_at = new Date();
  
      try {
          await query('INSERT INTO guestbook_comments (guestbook_id, mem_id, comment_text, created_at) VALUES (?, ?, ?, ?)', [guestbook_id, userId, content, created_at]);
          res.json({ success: true, message: '댓글 작성이 성공적으로 완료되었습니다.' });
      } catch (err) {
          console.error('댓글 작성 중 오류:', err);
          res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
      }
  });

// 방명록 댓글 삭제
router.delete('/delete/comment/:id', async function(req, res, next) {
    const guestbookcommentid = req.params.id;
    console.log("Delete Guestbook ID: " + guestbookcommentid);
    try {
        // 방명록 댓글 삭제 코드 작성
        await query('Delete from guestbook_comments where comment_id=? ', [guestbookcommentid]);
        res.status(200).json({ success: true, message: '방명록 댓글이 성공적으로 삭제되었습니다.' });
    } catch (error) {
        console.error('방명록 댓글 삭제 중 오류:', error);
        res.status(500).json({ success: false, message: '방명록 댓글 삭제 중 오류가 발생했습니다.' });
    }
});

module.exports = router;