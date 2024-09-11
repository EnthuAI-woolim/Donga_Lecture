class MiniroomDto {
    constructor(miniroomList) {
        this.imageUrl = miniroomList?.photo_url ?? null;
        this.content = miniroomList?.content ?? null;
    }

    // Getter와 Setter
    getImageUrl() {
        return this.imageUrl;
    }

    getContent() {
        return this.content;
    }
}

module.exports = MiniroomDto;