function clickUserProfile(id) {
    window.location = `/home/${id}`;
};

async function clickFriendFollow(id, follow) {
    const response = await fetch(`/friend/request/${id}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ follow: follow })
    });

    const result = await response.json();
    if (result.success) {
    }
    alert(result.message);
    window.location.reload();
}

async function friendList() {
    const id = document.getElementById('id').textContent;

    const response = await fetch(`/friend/friendList/${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const result = await response.json();
    console.log("========");
    if (result.success) {
        const friendListBox = document.getElementById('freind-page-list-box');
        friendListBox.innerHTML = result.friendListDto.map(friend => `
            <div class="friend-list-item" id="friend-list">
                <div class="friend-profile" onclick="clickUserProfile('${friend.id}')" style="background-image: url('${friend.imageUrl}');"></div>
                <div class="friend-id">${friend.id}</div>
                <div class="follow-btn" id="friendlist-follow-button" onclick="clickFriendFollow('${friend.id}', ${friend.follow})">
                    ${friend.follow ? '팔로잉' : '팔로우'}
                </div>
            </div>
        `).join('');
    } else {
        alert(result.message);
    }
}

async function friendRequestList() {
    const id = document.getElementById('id').textContent;

    const response = await fetch(`/friend/request/friendList/${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const result = await response.json();
    if (result.success) {
        const friendListBox = document.getElementById('freind-page-list-box');
        friendListBox.innerHTML = result.friendListDto.map(friend => `
            <div class="friend-list-item" id="friend-list">
                <div class="friend-profile" onclick="clickUserProfile('${friend.id}')" style="background-image: url('${friend.imageUrl}');"></div>
                <div class="friend-id">${friend.id}</div>
                <div class="follow-btn" id="friendlist-follow-button" onclick="clickFriendFollow('${friend.id}', ${friend.follow})">
                    ${friend.follow ? '팔로잉' : '팔로우'}
                </div>
            </div>
        `).join('');
    } else {
        alert(result.message);
    }
}


document.getElementById('friend-list-menu').addEventListener('click',  function() {
    const friendListMenu = document.getElementById('friend-list-menu');
    const friendListMenuText = document.getElementById('friend-list-menu-text');
    const friendRequestMenu = document.getElementById('friend-request-list-menu');
    const friendRequestMenuText = document.getElementById('friend-request-menu-text');

    if (friendListMenu && friendListMenuText && friendRequestMenu && friendRequestMenuText) {
        // 배경색과 글자색 변경
        friendListMenu.style.backgroundColor = '#f38f3d';
        friendListMenuText.style.color = 'white';

        // 다른 메뉴 스타일 초기화
        friendRequestMenu.style.backgroundColor = 'white';
        friendRequestMenuText.style.color = 'black';

        // friendList 함수 호출
        friendList();
    } else {
        console.error('하나 이상의 HTML 요소를 찾을 수 없습니다.');
    }
});
document.getElementById('friend-request-list-menu').addEventListener('click', function() {
    const friendListMenu = document.getElementById('friend-list-menu');
    const friendListMenuText = document.getElementById('friend-list-menu-text');
    const friendRequestMenu = document.getElementById('friend-request-list-menu');
    const friendRequestMenuText = document.getElementById('friend-request-menu-text');

    if (friendListMenu && friendListMenuText && friendRequestMenu && friendRequestMenuText) {
        // 배경색과 글자색 변경
        friendListMenu.style.backgroundColor = 'white';
        friendListMenuText.style.color = 'black';

        // 다른 메뉴 스타일 초기화
        friendRequestMenu.style.backgroundColor = '#f38f3d';
        friendRequestMenuText.style.color = 'white';

        // friendRequestList 함수 호출
        friendRequestList();
    } else {
        console.error('하나 이상의 HTML 요소를 찾을 수 없습니다.');
    }
});