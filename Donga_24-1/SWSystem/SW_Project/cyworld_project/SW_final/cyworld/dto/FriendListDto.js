class FriendListDto {
    constructor(friendList) {
        this.id = friendList?.friend_id ?? null;
        this.imageUrl = friendList?.profile_picture ?? 'https://cdn-icons-png.freepik.com/256/149/149071.png?semt=ais_hybrid';
        this.follow = friendList?.follow === 1 ? true : false;
    }

    // Getter와 Setter
    getId() {
        return this.id;
    }

    getImageUrl() {
        return this.imageUrl;
    }

    getFollow() {
        return this.follow;
    }
}

module.exports = FriendListDto;