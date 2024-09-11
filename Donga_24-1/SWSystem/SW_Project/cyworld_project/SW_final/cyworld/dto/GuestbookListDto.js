class GuestbookListDto {
    constructor(postList) {
        this.visitorId = postList?.visitor_id ?? null;
        this.content = postList?.content ?? null;
        this.date = postList?.created_at ?? null;
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

module.exports = GuestbookListDto;