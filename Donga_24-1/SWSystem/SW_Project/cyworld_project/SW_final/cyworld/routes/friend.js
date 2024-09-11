const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);

const verifyingUser = require('../public/middlewares/verifyingUser');
const layoutDto = require('../public/middlewares/layout');
const FriendListDto = require('../dto/FriendListDto');
const FollowDto = require('../dto/FollowDto');

// 친구 페이지 불러오기
router.get('/:id', verifyingUser, layoutDto, async function(req, res, next) {
    const id = req.params.id;
    const userId = req.user.userId;

    try {
        
        res.render('friend', { success: true, message: null, friendListDto: null });
    } catch (err) {
        console.error('친구 찾기 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 친구 목록 불러오기
router.get('/friendList/:id', verifyingUser, layoutDto, async function(req, res, next) {
    const id = req.params.id;
    const userId = req.user.userId;

    try {
        // 친구 목록
        const [ friendList ] = await query('CALL GetAllFriend(?, ?)', [userId, id]);
        const friendListDto = friendList.length > 0 ? friendList.map(item => new FriendListDto(item)) : [];
        
        res.json({ success: true, message: null, friendListDto: friendListDto });
    } catch (err) {
        console.error('친구 찾기 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 친구 요청 목록 
router.get('/request/friendList/:id', verifyingUser, layoutDto, async function(req, res, next) {
    const id = req.params.id;
    const userId = req.user.userId;

    try {
        // 친구 요청 목록
        const [requestList ] = await query('CALL GetRequestedFriend(?, ?)', [userId, id]);
        const friendListDto = requestList.length > 0 ? requestList.map(item => new FriendListDto(item)) : [];
        
        res.json({ success: true, message: null, friendListDto: friendListDto });
    } catch (err) {
        console.error('친구 찾기 중 오류 발생: ', err);
        res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// follower, following 불러오기
router.get('/follow', async function(req, res, next) {
    const userId = req.user.userId;
  
    try {
        const [[ followCount ]] = await query('CALL GetFollow(?)', [userId]);
        const followDto = new FollowDto(followCount);
        
        res.json({ success: true, message: '팔로우 불러오기 성공!', userId: userId, followDto: followDto });
    } catch (err) {
      
      res.json({ success: false, userId: userId, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
  
});

// 친구 요청 보내기
router.post('/request/:id', async function(req, res, next) {
    const id = req.params.id;
    const userId = req.user.userId;
    const { follow } = req.body;
    
    try {
        if (!follow) {  
            // 친구 요청  
            await query('INSERT INTO friendships (mem_id, friend_id) VALUES (?, ?)', [userId, id]);

            res.json({ success: true, message: '친구 요청을 보냈습니다!' });
        } else {
            // 친구 취소
            const [ existedfriendlist ] = await query('SELECT * FROM friendlist WHERE (mem_id = ? AND friend_id = ?) OR (mem_id = ? AND friend_id = ?)', [userId, id, id, userId]);
            if (existedfriendlist) {
                await query('DELETE FROM friendlist WHERE (mem_id = ? AND friend_id = ?) OR (mem_id = ? AND friend_id = ?)', [userId, id, id, userId]);
            }

            const [ existedfriendships ] = await query('SELECT * FROM friendships WHERE mem_id = ? AND friend_id = ?', [id, userId]);
            if (existedfriendships) {
                await query("UPDATE friendships SET status = 'pending' WHERE mem_id = ? AND friend_id = ?", [id, userId]);
            }

            await query('DELETE FROM friendships WHERE mem_id = ? AND friend_id = ?', [userId, id]);
    
            res.json({ success: true, message: '친구를 취소했습니다!' });
        }
    } catch (err) {
        res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }

});

// 친구 요청에 대한 응답하기(수락, 거절)
router.post('/response/:id', async function(req, res, next) {
    const id = req.params.id;
    const userId = req.user.userId;
    const { response } = req.body;
  
    try {
        if (response) {
            // 친구 요청 수락

            // friendships 테이블
            await query("UPDATE friendships SET status = 'accepted' WHERE mem_id = ? AND friend_id =?", [id, userId]);
            await query("INSERT INTO friendships (mem_id, friend_id, status) VALUES (?, ?, 'accepted')", [userId, id]);

            // friendlist 테이블
            await query('INSERT INTO friendlist (mem_id, friend_id) VALUES (?, ?)' [userId, id]);
            await query('INSERT INTO friendlist (mem_id, friend_id) VALUES (?, ?)' [id, userId]);

            res.json({ success: true, message: '친구 요청을 수락했습니다!' });
        } else {
            // 친구 요청 거절
            await query('DELETE FROM friendships WHERE mem_id = ? AND friend_id = ?', [id, userId]);

            res.json({ success: true, message: '친구 요청을 수락하지 않았습니다.' });
        }
    } catch (err) {
      
      res.json({ success: false, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
  
});

module.exports = router;