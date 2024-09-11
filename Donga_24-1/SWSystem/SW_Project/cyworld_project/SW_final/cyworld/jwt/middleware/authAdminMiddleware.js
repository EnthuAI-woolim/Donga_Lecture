const JwtAdminProvider = require('../provider/JwtAdminProvider');

function authAdminToken(req, res, next) {
    console.log('관리자 토큰 검증 중...'); // 디버깅 로그 추가
    const adminToken = req.cookies.adminToken;
    if (!adminToken) {
        console.log('관리자 토큰 없음');
        res.redirect('/admin/login');
        return res.status(403).send('관리자 토큰이 없습니다.');
    }

    // JWT 토큰 검증
    try {
        const decoded = JwtAdminProvider.verifyToken(adminToken);
        req.admin = decoded; // 검증된 토큰의 payload를 요청 객체에 저장하여 다음 미들웨어나 라우트에서 사용할 수 있도록 함
        next(); // 다음 미들웨어 호출
    } catch (error) {
        return res.sendStatus(403); // 토큰이 유효하지 않으면 Forbidden 에러 반환
    }
}
  
module.exports = authAdminToken;