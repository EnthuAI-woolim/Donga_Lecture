const db = require('../../db');
const util = require('util');

const query = util.promisify(db.query).bind(db);

const verifyingUser = async (req, res, next) => {
    const id = req.params.id; // URL에서 id를 가져옴
    const userId = req.user.userId;

    try {
        const [ user ] = await query('SELECT * from member where mem_id = ?', [id]);
        if (!user) return res.redirect(`/home/${userId}`);
    } catch (err) {
        console.error('데이터를 불러오는 중 오류 발생: ', err);
    }
    next();
};

module.exports = verifyingUser;
