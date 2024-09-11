const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);

const verifyingUser = require('../public/middlewares/verifyingUser');
const layoutDto = require('../public/middlewares/layout');
const photoListDto = require('../dto/PhotoListDto');


// photo 불러오기
router.get('/:id', verifyingUser, layoutDto, async function(req, res, next) {
  const id = req.params.id;

  try {
      const photoListDto = await query('SELECT title, content, photo_url, photo_id FROM photo WHERE mem_id = ? ORDER BY created_at DESC', [id]);

      // const photoList = await query('SELECT title, content, photo_url FROM photo WHERE mem_id = ?', [id]);
      // const photoListDto = photoList.length > 0 ? photoList.map(item => new PhotoListDto(item)) : [new PhotoListDto()];

      res.render('photo', { success: true, id: id, message: null, photoListDto: photoListDto });
  } catch (err) {
      console.error('타이틀 변경 중 오류 발생: ', err);
      res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
  }
});

// 작성한 사진첩 저장하기
router.post('/write/:id', async function(req, res, next) {
  const userId = req.user.userId;
  const { title, content, imageUrl } = req.body;
  try {
      await query('INSERT INTO photo (mem_id, title, content, photo_url) VALUES (?, ?, ?, ?)', [userId, title, content, imageUrl]);

      res.json({ success: true, id: userId, message: '사진첩 작성 성공!' });
  } catch (err) {
      res.json({ success: false, id: userId, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
  }
});

router.delete('/delete/:id', async function(req, res, next) {
  const photoId = req.params.id;
  console.log("Delete Photo ID: " + photoId); // 삭제할 사진의 ID 확인
  try {
    await query('DELETE FROM photo WHERE photo_id = ?', [photoId]);
    res.status(200).json({ success: true, message: '사진이 성공적으로 삭제되었습니다.'});
  } catch (error) {
    console.error('사진 삭제 중 오류:', error);
    res.status(500).json({ success: false, message: '사진 삭제 중 오류가 발생했습니다.'})
  }
});


module.exports = router;