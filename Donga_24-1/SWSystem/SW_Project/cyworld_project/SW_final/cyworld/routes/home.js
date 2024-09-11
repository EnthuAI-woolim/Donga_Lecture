const express = require('express');
const router = express.Router();
const db = require('../db');
const util = require('util');
const query = util.promisify(db.query).bind(db);
const multer = require('multer');
const path = require('path');
const fs = require('fs');


const verifyingUser = require('../public/middlewares/verifyingUser');
const layoutDto = require('../public/middlewares/layout');
const MainhomeDto = require('../dto/MainhomeDto');
const MiniroomDto = require('../dto/MiniroomDto');
const UpdatednewsDto = require('../dto/UpdatednewsDto');

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

const upload = multer({ storage: storage });


// 홈 화면 불러오기
router.get('/:id', verifyingUser, layoutDto, async function(req, res, next) {
    const id = req.params.id;       // 현재 페이지의 유저 ID
    
    try {
        // description: 프로시저로 실행
        // updatednewsDto.getUpdatedList()[0].getTableName()
        const [ updatedList ] = await query('CALL GetUpdatedPostPhoto(?)', [id]);
        const [[ updatedCount ]] = await query('CALL GetCount(?)', [id]);
        const updatednewsDto = new UpdatednewsDto(updatedList, updatedCount);


        // description: 미니룸 정보 가져오기
        // miniroomDto.getMemo()
        const [ miniroomList ] = await query('SELECT photo_url, content FROM miniroom WHERE mem_id = ?', [id]);
        const miniroomDto = new MiniroomDto(miniroomList);

        const [ result ] = await query('SELECT photo_url, content FROM miniroom WHERE mem_id = ?', [id]);
        const [ nickname ] = await query('SELECT photo_url, content FROM miniroom WHERE mem_id = ?', [id]);
        const mainhomeDto = new MainhomeDto(result, nickname);



        //            render            //
        res.render('home', { id: id, updatednewsDto: updatednewsDto, miniroomDto: miniroomDto });
    } catch (err) {
        console.error('데이터를 불러오는 중 오류 발생: ', err);
        res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});

// 미니룸 변경하기
router.patch('/edit_miniroom/:id', verifyingUser, layoutDto, upload.single('miniroomImage'), async function(req, res, next) {
    const id = req.params.id;

    // 이미지 파일이 존재하는지 확인
    if (!req.file) {
        return res.json({ success: false, id: id, message: '이미지 파일이 업로드되지 않았습니다.' });
    }

    const imageUrl = `/images/${req.file.filename}`; // 업로드된 이미지의 서버 상의 경로


    try {
        await query('UPDATE miniroom SET photo_url = ? WHERE mem_id = ?', [ imageUrl, id]);

        res.json({ success: true, id: id, imageUrl: imageUrl, message: '미니룸 변경 성공!' });
    } catch (err) {
        console.error('타이틀 변경 중 오류 발생: ', err);
        res.json({ success: false, id: id, message: '서버 오류 발생. 나중에 다시 시도하세요.' });
    }
});




module.exports = router;