const jwt = require('jsonwebtoken');
const { SECRET_KEY } = require('../configs/config');

// 사용자 토큰 생성
class JwtProvider {
    // 사용자의 아이디와 비밀번호를 확인하여 유효한 경우 JWT 토큰을 생성하는 함수
    static generateToken(id) {
        // JWT에 포함될 payload 데이터 정의 (여기서는 사용자의 아이디만 포함)
        const payload = {
            userId: id
        };

        // JWT 토큰 생성 및 반환 (여기서는 간단한 예시로 시크릿 키를 사용하여 서명)
        const token = jwt.sign(payload, SECRET_KEY, { expiresIn: '1h' }); // 토큰 만료 시간: 1시간
        return token;
    }

    // JWT 토큰을 검증하는 함수
    static verifyToken(token) {
        try {
            // 토큰을 검증하고 검증된 payload를 반환
            const decoded = jwt.verify(token, SECRET_KEY);
            console.log('토큰 검증 성공!');
            return decoded;
        } catch (err) {
            // 토큰 검증 실패 시 에러를 반환하거나 처리
            console.error('토큰 검증 실패: ', err);
            return null;
        }
    }
}

module.exports = JwtProvider;
