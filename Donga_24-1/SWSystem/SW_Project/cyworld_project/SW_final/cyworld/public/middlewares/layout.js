const db = require('../../db');
const util = require('util');

const MainhomeDto = require('../../dto/MainhomeDto');
const MiniroomDto = require('../../dto/MiniroomDto');

const query = util.promisify(db.query).bind(db);

const layoutData = async (req, res, next) => {
    const id = req.params.id; // URL에서 id를 가져옴
    const userId = req.user.userId;
    var myPage = 'false';

    try {
        // 요청된 페이지가 사용자의 페이지인지 확인
        if (id === userId) myPage = 'true';

        // 홈에서만 count_today, count_total의 값 증가
        if (req.originalUrl.startsWith('/home/')) 
            await query('UPDATE mainhome SET count_today = count_today + 1, count_total = count_total + 1 WHERE mem_id = ?', [id]);
        
        //              select            // 
        // description: mainhomeDto.getToday() - 함수를 통해 데이터 가져올 수 있음
        const [ nickname ] = await query('SELECT nickname from member where mem_id = ?', [id]);
        const [ mainhomeResult ] = await query('SELECT profile_picture, profile_bio, count_today, count_total, title FROM mainhome WHERE mem_id = ?', [id]);
        const mainhomeDto = new MainhomeDto(mainhomeResult, nickname);

        const [ miniroomResults ] = await query('SELECT * FROM miniroom WHERE mem_id = ?', [id]);
        const miniroomDto = new MiniroomDto(miniroomResults);
        
        res.locals.id = id;
        res.locals.userId = userId;
        res.locals.myPage = myPage;
        res.locals.mainhomeDto = mainhomeDto;
        res.locals.miniroomDto = miniroomDto;
        res.locals.userListDto = [];
    } catch (err) {
        console.error('데이터를 불러오는 중 오류 발생: ', err);
    }
    next();
};

module.exports = layoutData;
