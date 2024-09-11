


// 모달 열기
var modal = document.getElementById("reportModal");
var btn = document.getElementById("reportButton");
var span = document.getElementsByClassName("report-modal-close")[0];

btn.onclick = function() {
    modal.style.display = "flex";
}

// 모달 닫기
span.onclick = function() {
    modal.style.display = "none";
}

// 모달 외부 클릭 시 닫기
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

async function reportUser(event) {
    event.preventDefault(); // 폼의 기본 제출 동작을 막음

    const id = document.getElementById('user-id').value;
    const reason = document.getElementById('reason').value;

    
    const response = await fetch(`/setting/member_report/${id}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ reason })
    });

    const result = await response.json();
    alert(result.message);
    window.location.reload();
    
};


async function member_getout(event) {
    event.preventDefault(); // 폼 제출을 막음
    const userId = document.getElementById('userIdDiv').textContent;
    console.log(userId);
    
    const confirmDelete = confirm('정말로 회원 탈퇴하시겠습니까?');

    if (confirmDelete) {
    const response = await fetch(`/setting/member_getout/${userId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });

        const result = await response.json();
        alert(result.message);
        
    } else {
        console.log('회원 탈퇴 취소');
    }

    window.location.reload();

};

document.addEventListener('DOMContentLoaded', () => {
    const deleteButton = document.getElementById('delete-button');
    const reportButton = document.getElementById('reportForm');

    if (deleteButton) deleteButton.addEventListener('click', member_getout);
    if (reportButton) reportButton.addEventListener('submit', reportUser);
        
});
