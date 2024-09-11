const moment = require('moment-timezone');

class MemberListDto {
    constructor(memberList) {
        this.id = memberList?.mem_id;
        this.password = memberList?.mem_pw;
        this.nickname = memberList?.nickname;
        this.date = moment.tz(memberList?.created_at, 'Asia/Seoul').format('YYYY-MM-DD HH:mm:ss');
    }

    // Getter와 Setter
    getId() {
        return this.id;
    }

    getPassword() {
        return this.password;
    }

    getNickname() {
        return this.nickname;
    }

    getDate() {
        return this.date;
    }
}

module.exports = MemberListDto;