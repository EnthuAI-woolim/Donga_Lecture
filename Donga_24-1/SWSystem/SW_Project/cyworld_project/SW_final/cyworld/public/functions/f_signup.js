async function signupHandler(event) {
    event.preventDefault(); // 폼 제출을 막음

    const id = document.getElementById('id').value;
    const password = document.getElementById('password').value;
    const checkPassword = document.getElementById('checkPassword').value;
    const nickname = document.getElementById('nickname').value;

    if (password !== checkPassword) {
        alert('비밀번호가 일치하지 않습니다.');
        window.location.href = `/auth/signup`;
        return;
    }

    const response = await fetch('/auth/signup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ id, password, nickname })
    });

    const result = await response.json();

    alert(result.message);
    if (result.success) window.location.href = `/auth/login`;
    else window.location.href = `/auth/signup`;
}

function loginHandler(event) {
    window.location.href = '/auth/login';
}

// DOM 요소가 로드되었을 때 이벤트 리스너를 추가합니다.
document.addEventListener('DOMContentLoaded', () => {
    const signupButton = document.getElementById('signup-button');
    const loginButton = document.getElementById('login-button');

    if (signupButton) signupButton.addEventListener('click', signupHandler);
    if (loginButton) loginButton.addEventListener('click', loginHandler);
});