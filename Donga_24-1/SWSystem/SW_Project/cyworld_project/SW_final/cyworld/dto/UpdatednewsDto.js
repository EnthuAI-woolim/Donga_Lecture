const UpdatedList = require('./component/UpdatedList');

class UpdatednewsDto {
    constructor(updatedList, updatedCount) {
        this.updatedList = updatedList && updatedList.length > 0 ? updatedList.map(item => new UpdatedList(item)) : [];
    
        this.postCount = updatedCount.post_count;
        this.photoCount = updatedCount.photo_count;
        this.guestbookCount = updatedCount.guestbook_count;
        this.totalCount = this.postCount + this.photoCount + this.guestbookCount;
    }

// toString 메서드 정의 (현재 클래스의 경우 필요시 정의)
toString() {
    return JSON.stringify({
        updatedList: this.updatedList.map(item => item.toString()), // 각 UpdatedList 객체를 문자열로 변환
    });
}

// Getter와 Setter
getUpdatedList() {
    return this.updatedList;
}

getPostCount() {
    return this.postCount;
}

getPhotoCount() {
    return this.photoCount;
}

getGuestbookCount() {
    return this.guestbookCount;
}

getTotalCount() {
    return this.totalCount;
}

 // 새로운 메서드 추가
 getTableNames(num) {
    if (this.updatedList.length === 0) {
        return null; // 데이터가 없을 경우 null 반환
    }

    if (this.updatedList.length <= num ) {
        return []; // 데이터가 없을 경우 null 반환
    }

    // 데이터를 최신 업데이트 순으로 정렬 (가정)
    const sortedList = this.updatedList.sort((a, b) => {
        // 가정: updatedTime이라는 속성을 기준으로 정렬한다고 가정
        // 만약 updatedTime이 없는 경우에는 다른 기준으로 정렬해야 함
        return b.post_id - a.post_id;
    });

    // 정렬된 리스트에서 첫 번째 요소의 tableName 반환
    return sortedList[num].getTableName();
}

getTitles(num) {
    if (this.updatedList.length === 0) {
        return []; // 데이터가 없을 경우 null 반환
    }

    if (this.updatedList.length <= num ) {
        return []; // 데이터가 없을 경우 null 반환
    }

    // 데이터를 최신 업데이트 순으로 정렬 (가정)
    const sortedList = this.updatedList.sort((a, b) => {
        // 가정: updatedTime이라는 속성을 기준으로 정렬한다고 가정
        // 만약 updatedTime이 없는 경우에는 다른 기준으로 정렬해야 함
        return b.post_id - a.post_id;
    });

    // 정렬된 리스트에서 첫 번째 요소의 title 반환
    return sortedList[num].getTitle();
}
}

module.exports = UpdatednewsDto;