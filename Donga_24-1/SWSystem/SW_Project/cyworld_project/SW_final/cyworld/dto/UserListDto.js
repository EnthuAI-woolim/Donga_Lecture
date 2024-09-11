class UserListDto {
    constructor(userList) {
        this.mem_id = userList?.mem_id ?? null;
        this.profileImage = userList?.profile_picture ?? 'https://cdn-icons-png.freepik.com/256/149/149071.png?semt=ais_hybrid';
        this.follow = userList?.follow === 1 ? true : false;
    }

    // Getter와 Setter
    getMem_id() {
        return this.mem_id;
    }

    getProfileImage() {
        return this.profileImage;
    }

    getFollow() {
        return this.follow;
    }
}

module.exports = UserListDto;