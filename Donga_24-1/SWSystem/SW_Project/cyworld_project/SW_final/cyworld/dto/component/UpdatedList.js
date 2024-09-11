class UpdatedList {
    constructor(updatedList) {
        this.tableName = updatedList?.table_name ?? null;
        this.title = updatedList?.title ?? null;
    }

    // toString 메서드 정의
    toString() {
        return JSON.stringify({
            tableName: this.tableName,
            title: this.title
        }); // 객체를 JSON 문자열로 반환
    }

    // Getter와 Setter
    getTableName() {
        return this.tableName;
    }

    getTitle() {
        return this.title;
    }

}

module.exports = UpdatedList;