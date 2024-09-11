class PhotoListDto {
    constructor(postList) {
        this.title = postList?.title ?? null;
        this.content = postList?.content ?? null;
        this.imageUrl = postList?.photo_url ?? null;
    }

    // Getter와 Setter
    getTitle() {
        return this.title;
    }

    getContent() {
        return this.content;
    }

    getPost_date() {
        return this.post_date;
    }
}

module.exports = PhotoListDto;