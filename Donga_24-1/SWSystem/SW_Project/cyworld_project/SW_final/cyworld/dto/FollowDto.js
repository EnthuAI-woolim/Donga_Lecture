class FollowDto {
    constructor(followCount) {
        this.follower = followCount.follower;
        this.following = followCount.following;
    }

    // Getter와 Setter
    getFollower() {
        return this.follower;
    }

    getFollowing() {
        return this.following;
    }
}

module.exports = FollowDto;