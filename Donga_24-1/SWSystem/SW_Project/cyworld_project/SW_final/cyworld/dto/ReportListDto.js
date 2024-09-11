const moment = require('moment-timezone');

class ReportListDto {
    constructor(reportList) {
        this.reporter_id = reportList?.reporter_id;
        this.reported_id = reportList?.reported_id;
        this.reason = reportList?.reason ?? '신고 사유 없음';
        this.date = moment.tz(reportList?.report_time, 'Asia/Seoul').format('YYYY-MM-DD HH:mm:ss');
    }

    // Getter와 Setter
    getReporterId() {
        return this.reporter_id;
    }

    getReprotedId() {
        return this.reported_id;
    }

    getReason() {
        return this.reason;
    }

    getDate() {
        return this.date;
    }
}

module.exports = ReportListDto;