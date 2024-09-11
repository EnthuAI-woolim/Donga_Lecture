class MainhomeDto {
    constructor(result, nickname) {
        this.imageUrl = result?.profile_picture ?? 'https://cdn-icons-png.freepik.com/256/149/149071.png?semt=ais_hybrid';
        this.intro = result?.profile_bio ?? '나를 소개해보세요.';
        this.today = result?.count_today;
        this.total = result?.count_total;
        this.title = result?.title ?? '타이틀을 작성해보세요.';
        this.nickname = nickname?.nickname;
    }

    // Getter와 Setter
    getImageUrl() {
        return this.imageUrl;
    }

    getIntro() {
        return this.intro;
    }

    getToday() {
        return this.today;
    }

    getTotal() {
        return this.total;
    }

    getTitle() {
        return this.title;
    }

    getNickname() {
        return this.nickname;
    }

}

module.exports = MainhomeDto;