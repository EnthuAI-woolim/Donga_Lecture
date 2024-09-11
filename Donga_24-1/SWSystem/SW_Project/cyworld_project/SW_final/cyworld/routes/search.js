const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);

const UserListDto = require('../dto/UserListDto');


// 전체 유저 불러오기
router.get('/user', async function(req, res, next) {
    const userId = req.user.userId;

    try {
        const [ userList ] = await query('CALL GetAllUser(?)', [userId]);
        const userListDto = userList.length > 0 ? userList.map(item => new UserListDto(item)) : [];
        
        res.json({ success: true, message: null, userListDto: userListDto });
    } catch (err) {
        console.error('전체 유저 찾기 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 특정 유저 찾기
router.get('/user/:id', async function(req, res, next) {
    const id = req.params.id;
    const userId = req.user.userId;

    try {
        const [ existingUsers ] = await query('SELECT * FROM member WHERE mem_id = ?', [id]);
        if (!existingUsers) {
            return res.json({ success: false, message: '존재하지 않는 사용자입니다.' });
        }

        const [ userList ] = await query('CALL GetUser(?, ?)', [ userId, `%${id}%`]);
        const userListDto = userList.length > 0 ? userList.map(item => new UserListDto(item)) : [];

        res.json({ success: true, id: id, userListDto: userListDto });
    } catch (err) {
        console.error('특정 유저 찾기 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});



module.exports = router;