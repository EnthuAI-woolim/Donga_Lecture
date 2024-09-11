// 페이지 로드 시 버튼 숨기기
document.addEventListener('DOMContentLoaded', function() {
  const { myPage } = window.config;

  if (myPage === 'false') {
      document.getElementById('openImageModal').style.display = 'none';
  }
  if (myPage === 'false') {
    const deleteButtons = document.querySelectorAll('.photo-delete-button');
    deleteButtons.forEach(button => {
      button.style.display = 'none';
    });
  }
});

// 사진 등록 모달 열기
function openImageModal() {
  var modal = document.getElementById('modal');
  modal.style.display = 'block';
}

// 사진 등록 모달 닫기
function closeModal() {
  var modal = document.getElementById('modal');
  modal.style.display = 'none';
}

// 사진 등록 폼 제출 처리
function submitPhoto() {
  console.log('submitPhoto 함수 호출');

  const imageUrl = document.getElementById('imageUrl').value;
  const title = document.getElementById('title').value;
  const content = document.getElementById('content').value;
  const { userId } = window.config;

  // 입력값 검증
  if (!imageUrl || !title || !content) {
    alert('이미지 URL, 제목, 내용을 모두 입력하세요.');
    return;
  }

  // 서버로 데이터 전송
  fetch(`/photo/write/${userId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      title: title,
      content: content,
      imageUrl: imageUrl
    })
  })
    .then(response => response.json())
    .then(data => {
      console.log('submitPhoto 데이터 검증 시작');
      if (data.success) {
        alert('사진첩 작성 성공!');
        // 성공적으로 작성되면 모달 닫기
        closeModal();
        // 페이지 리로드 또는 다시 렌더링하여 새로운 사진 첩을 표시
        location.reload();
      } else {
        alert('서버 오류 발생. 나중에 다시 시도하세요.');
      }
    })
    .catch(error => {
      console.error('사진첩 작성 중 오류:', error);
      alert('사진첩 작성 중 오류가 발생했습니다.');
    });
}

// X 버튼 클릭 시 모달 닫기
document.getElementById('modal-close').addEventListener('click', function () {
  closeModal();
});

// 사진 등록 버튼 클릭 이벤트 처리
document.getElementById('openImageModal').addEventListener('click', function () {
  openImageModal();
});

// 사진 등록 폼 제출 이벤트 처리
document.getElementById('write').addEventListener('click', function () {
  submitPhoto();
});

// 사진 삭제 함수
function deleteImage(event) {
  // 우클릭 메뉴가 뜨는 것을 방지하고 이벤트 기본 동작 중지
  event.preventDefault();

  // "사진을 삭제하겠습니까?" 경고창 표시
  if (confirm('사진을 삭제하겠습니까?')) {
    // 확인 버튼을 누르면 사진 삭제
    event.target.remove();

    // 사진이 삭제된 후에 등록된 사진이 없을 때 메시지 표시
    if (document.querySelectorAll('.content-photo-item').length === 0) {
      document.querySelector('.no-image-msg').style.display = 'block';
    }
  }
}

// 이미지 우클릭 이벤트 처리
document.getElementById('photoGallery').addEventListener('contextmenu', function (event) {
  if (event.target && event.target.nodeName === 'IMG') {
    deleteImage(event);
  }
});

// 이미지 클릭 시 모달 열기
function openModal(event) {
  var modal = document.getElementById('modal');
  var modalImg = document.getElementById('modal-img');
  var captionText = document.getElementById('caption');

  modal.style.display = 'block';
  modalImg.src = event.target.src;
  captionText.innerHTML = event.target.alt;
}

// 모달 닫기
document.getElementById('modal-close').addEventListener('click', function () {
  closeModal();
});

// 이미지 클릭 시 모달 열기
document.getElementById('photoGallery').addEventListener('click', function (event) {
  if (event.target && event.target.nodeName === 'IMG') {
    openModal(event);
  }
});

// 댓글 입력 이벤트 처리
document.getElementById('comment-submit').addEventListener('click', function () {
  submitComment();
});

// 페이지 로드시 등록된 사진이 없다면 메시지 표시
// if (document.querySelectorAll('.content-photo-item').length === 0) {
//   document.querySelector('.no-image-msg').style.display = 'block';
// }

function deletePhoto(photoId) {
  fetch(`/photo/delete/${photoId}`, {
    method: 'DELETE'
  })
    .then(response => response.json())
    .then(data => {
      console.log('서버 응답:', data);
      if (data.success) {
        alert('사진이 성공적으로 삭제되었습니다.');
        location.reload();
      } else {
        alert('사진 삭제 중 오류가 발생했습니다.');
      }
    })
    .catch(error => {
      console.error('사진 삭제 중 오류:', error);
      alert('사진 삭제 중 오류가 발생했습니다.');
    });
}