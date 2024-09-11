class PostListDto {
    constructor(postList) {
        this.title = postList?.title ?? null;
        this.content = postList?.content ?? null;
        this.post_date = postList?.post_date ?? null;
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

module.exports = PostListDto;