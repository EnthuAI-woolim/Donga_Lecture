// 페이지 로드 시 버튼 숨기기
document.addEventListener('DOMContentLoaded', function() {
    const { myPage } = window.config;
    const userId = document.getElementById('userId').textContent;
    if (myPage === 'false') {
        const deleteButtons = document.querySelectorAll('.guestbook-delete-button');
        deleteButtons.forEach(button => {
          button.style.display = 'none';
        });
      }

    if (myPage === 'false') {
        const deleteButtons = document.querySelectorAll('.guestbook-comment-delete-button');
        deleteButtons.forEach(button => {
          button.style.display = 'none';
        });
      }
  });

// 모달 열기
var modal = document.getElementById("guestbook-modal");
var btn = document.getElementById("guestbook-open-btn");
var span = document.getElementsByClassName("guestbook-modal-close")[0];

btn.onclick = function() {
    modal.style.display = "block";
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

// 작성자 아이디와 내용을 전송
async function submitGuestbook() {
    // 입력된 값 가져오기
    const id = document.getElementById('idDiv').textContent;
    const content = document.getElementById('content').value;

    // 입력 값 확인
    if (!content) {
        alert('내용을 입력하세요.');
        return;
    }

    // 서버로 데이터 전송
    try {
        const response = await fetch(`/guestbook/write/${id}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ content })
        });

        const result = await response.json();

        if (response.ok) {
            alert(result.message);
            // 페이지 새로고침
            window.location.reload();
        } else {
            console.error('서버 응답 오류:', result);
            alert('방명록 작성 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('fetch 요청 중 오류:', error);
        alert('서버와의 통신 중 오류가 발생했습니다.');
    }
}

function toggleCommentForm(guestbookId) {
    var form = document.getElementById('comment-form-' + guestbookId);
    if (form.style.display === 'none' || form.style.display === '') {
        form.style.display = 'block';
    } else {
        form.style.display = 'none';
    }
}

async function submitComment(guestbookId) {
    var content = document.getElementById('comment_content_' + guestbookId).value;

    if (!content) {
        alert('댓글 내용을 입력하세요.');
        return;
    }

    // 댓글 데이터를 서버에 제출하는 코드
    try {
        const response = await fetch('/guestbook/submit_comment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                guestbook_id: guestbookId,
                content: content
            })
        });

        const data = await response.json();

        if (data.success) {
            alert('댓글이 성공적으로 작성되었습니다.');
            location.reload();  // 페이지를 새로고침하여 댓글을 표시합니다.
        } else {
            alert('댓글 작성에 실패했습니다.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('서버와의 통신 중 오류가 발생했습니다.');
    }
}

function deleteGuestbook(guestbookId) {
    // AJAX를 사용하여 서버로 삭제 요청을 보냅니다.
    fetch(`/guestbook/delete/${guestbookId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('방명록가 성공적으로 삭제되었습니다.');
            // 성공적으로 삭제되면 페이지를 새로고침하여 업데이트된 방명록 리스트를 보여줍니다.
            location.reload();
        } else {
            alert('방명록 삭제 중 오류가 발생했습니다.');
        }
    })
    .catch(error => {
        console.error('방명록 삭제 중 오류:', error);
        alert('방명록 삭제 중 오류가 발생했습니다.');
    });
}

function deleteGuestbookcomment(guestbookId) {
    // AJAX를 사용하여 서버로 삭제 요청을 보냅니다.
    fetch(`/guestbook/delete/comment/${guestbookId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('방명록 댓글이 성공적으로 삭제되었습니다.');
            // 성공적으로 삭제되면 페이지를 새로고침하여 업데이트된 방명록 댓글 리스트를 보여줍니다.
            location.reload();
        } else {
            alert('방명록 댓글 삭제 중 오류가 발생했습니다.');
        }
    })
    .catch(error => {
        console.error('방명록 댓글 삭제 중 오류:', error);
        alert('방명록 댓글 삭제 중 오류가 발생했습니다.');
    });
}