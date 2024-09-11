const id = document.getElementById('layout-id').textContent;
const userId = document.getElementById('layout-userId').textContent;
const myPage = document.getElementById('layout-myPage').textContent;

// 설정 버튼 숨기기
if (myPage === 'false') {
    document.getElementById('setting-button').style.display = 'none';
    document.getElementById('profile-edit').style.display = 'none';
    document.getElementById('profile-text-edit').style.display = 'none';
    document.getElementById('nickname-edit').style.display = 'none';
    document.getElementById('title-edit').style.display = 'none';
}


document.addEventListener('DOMContentLoaded', () => {
    const searchButton = document.getElementById('search-button');
    const searchIdButton = document.getElementById('search-id-button');
    const closeSearchButton = document.getElementById('closeSearchModal');
    const myhomeButton = document.getElementById('myhome-button');
    const logoutButton = document.getElementById('logout-button');
    const profileButton = document.getElementById('profile-edit');
    const introButton = document.getElementById('profile-text-edit');
    const nicknameButton = document.getElementById('nickname-edit');
    const titleButton = document.getElementById('title-edit');
    const imageUpload = document.getElementById('profile-input'); // 프로필 입력 필드
    const profileContainer = document.getElementById('profile-image-container'); // 이미지 컨테이너

    searchButton?.addEventListener('click', () => {
        document.getElementById('searchModal').style.display = 'block';
    });

    searchIdButton?.addEventListener('click', () => {
        document.getElementById('userScrollbox').style.display = '';
        searchId();
    });

    closeSearchButton?.addEventListener('click', function() {
        document.getElementById('searchModal').style.display = 'none';
        document.getElementById('search-id').value = '';
        document.getElementById('userScrollbox').innerHTML = '';
        document.getElementById('userScrollbox').style.display = 'none';
    });

    myhomeButton?.addEventListener('click', () => {
        window.location.href = `/home/${userId}`;
    });

    logoutButton?.addEventListener('click', logout);

    profileButton?.addEventListener('click', () => {
        document.getElementById('profile-input').click();
    });

    introButton?.addEventListener('click', () => {
        layout_toggleEdit(introButton, document.getElementById('profile-text-display'), document.getElementById('profile-text-input'), 'intro');
    });

    nicknameButton?.addEventListener('click', () => {
        layout_toggleEdit(nicknameButton, document.getElementById('nickname-display'), document.getElementById('nickname-input'), 'nickname');
    });

    titleButton?.addEventListener('click', () => {
        layout_toggleEdit(titleButton, document.getElementById('title-display'), document.getElementById('title-input'), 'title');
    });
});

async function searchId() {
    const search_id = document.getElementById('search-id').value;
    const url = search_id ? `/search/user/${search_id}` : '/search/user';

    const response = await fetch(url, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const result = await response.json();

    const userScrollbox = document.getElementById('userScrollbox');
    userScrollbox.innerHTML = '';

    if (result.success) {
        result.userListDto.forEach(user => {
            const userDiv = document.createElement('div');
            userDiv.className = 'user-list-item';

            const profileDiv = document.createElement('div');
            profileDiv.className = 'user-profile';
            profileDiv.style.backgroundImage = `url('${user.profileImage}')`;

            const idDiv = document.createElement('div');
            idDiv.className = 'user-id';
            idDiv.textContent = user.mem_id;

            const followBtn = document.createElement('div');
            followBtn.className = 'follow-btn';
            followBtn.textContent = user.follow ? '팔로잉' : '팔로우';

            userDiv.appendChild(profileDiv);
            userDiv.appendChild(idDiv);
            userDiv.appendChild(followBtn);

            userScrollbox.appendChild(userDiv);

            // 각 userDiv에 클릭 이벤트 추가
            profileDiv.addEventListener('click', () => {
                const searchId = idDiv.textContent;
                clickUser(searchId);
            });

            followBtn.addEventListener('click', () => {
                const searchId = idDiv.textContent;
                const follow = user.follow;
                clickFollow(searchId, follow);

                user.follow = !user.follow;
                followBtn.textContent = user.follow ? '팔로잉' : '팔로우';
            });

        });
    } else {
        const errorMessage = document.createElement('p');
        errorMessage.textContent = '사용자를 찾을 수 없습니다.';
        userScrollbox.appendChild(errorMessage);
    }
}

async function clickUser(id) {
    window.location = `/home/${id}`;
}

async function clickFollow(id, follow) {
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
}

async function logout() {
    const response = await fetch('/auth/logout', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify()
    });

    const result = await response.json();

    alert(result.message);
    window.location.reload();
}

async function layout_toggleEdit(button, displayElement, inputElement, valueName) {
    if (inputElement.style.display === 'none' || inputElement.style.display === '') {
        inputElement.value = displayElement.innerText;
        displayElement.style.display = 'none';
        inputElement.style.display = 'inline';
        inputElement.focus();
        button.innerText = 'save';
    } else {
        const value = inputElement.value;
        const data = { [valueName]: value };

        const response = await fetch(`/layout/edit_${valueName}/${userId}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        alert(result.message);
        window.location.reload();
    }
}

imageUpload.addEventListener('change', async function(event) {
    const file = event.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('profileImage', file);
        formData.append('id', userId);  // 사용자 ID 전송

        // console.log(file);
  
        try {
          const response = await fetch(`/home/profile-edit/${userId}`, {
            method: 'PATCH',
            body: formData
          });
          
         
          const result = await response.json();
          if (result.success) {

            // 이미지를 화면에 표시하는 코드는 여기에서 추가
            // 이미지의 URL은 result.imageUrl에서 가져와서 사용
            const img = document.createElement('img');
            img.src = result.imageUrl;
            img.classList.add('profile-image-img'); // 이미지에 클래스 추가 (선택 사항)
            profileContainer.innerHTML = ''; // 이미지 컨테이너 내용 비우기
            profileContainer.appendChild(img);
          } else {
            alert('이미지 업로드 실패: ' + result.message);
          }
        } catch (err) {
          console.error('이미지 업로드 중 오류:', err);
          alert('이미지 업로드 중 오류가 발생했습니다.');
             }
        }
    });

