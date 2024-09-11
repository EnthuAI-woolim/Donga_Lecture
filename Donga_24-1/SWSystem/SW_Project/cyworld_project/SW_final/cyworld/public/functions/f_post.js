// 페이지 로드 시 버튼 숨기기
document.addEventListener('DOMContentLoaded', function() {
  const { myPage } = window.config;

  if (myPage === 'false') {
    document.getElementById('openDiaryModal').style.display = 'none';
  }
  if (myPage === 'false') {
    const deleteButtons = document.querySelectorAll('.post-delete-button');
    deleteButtons.forEach(button => {
      button.style.display = 'none';
    });
  }
});

// 다이어리 작성 버튼 클릭 시 모달 열기
  document.getElementById('openDiaryModal').addEventListener('click', function() {
    document.getElementById('diaryModal').style.display = 'block';
  });

  // 모달 닫기 버튼 클릭 시 모달 닫기
  document.getElementById('closeDiaryModal').addEventListener('click', function() {
    document.getElementById('diaryModal').style.display = 'none';
  });

  // 다이어리 작성 폼 제출 시 처리
  document.getElementById('diaryForm').addEventListener('submit', function(event) {
    event.preventDefault(); // 기본 동작 방지

    // 입력된 값 가져오기
    var diarytitle = document.getElementById('diarytitle').value;
    var diaryDate = document.getElementById('diaryDate').value;
    var diaryContent = document.getElementById('diaryContent').value;

    // 다이어리 요소 생성
    var diaryEntry = document.createElement('div');
    diaryEntry.className = 'diary';
    diaryEntry.innerHTML =  '<div class="diary-title">' + diarytitle + '</div>' +
                            '<div class="diary-date">' + diaryDate + '</div>' +
                           '<div class="diary-contents"><p>' + diaryContent + '</p></div>';

    // 다이어리 스크롤박스에 추가
    var diaryScrollbox = document.getElementById('diaryScrollbox');
    var firstDiary = diaryScrollbox.firstElementChild;
    // 다이어리 스크롤박스에 자식이 있으면 첫 번째 자식 앞에 새로운 다이어리 추가
    if (firstDiary) {
      diaryScrollbox.insertBefore(diaryEntry, firstDiary);
    } else {
      // 다이어리 스크롤박스에 자식이 없으면 마지막에 추가
      diaryScrollbox.appendChild(diaryEntry);
    }

    // 작성 모달 닫기
    document.getElementById('diaryModal').style.display = 'none';

    // 작성된 내용 초기화

    document.getElementById('diarytitle').value = '';
    document.getElementById('diaryDate').value = '';
    document.getElementById('diaryContent').value = '';
  });

  
  // 달력 플래너 기능 추가 (간단한 달력 예시)
  document.addEventListener('DOMContentLoaded', function() {
    var plannerCalendar = document.getElementById('plannerCalendar');
    var today = new Date();
    var year = today.getFullYear();
    var month = today.getMonth() + 1;
  
    var calendarHtml = '<table class="calendar-table"><tr>';
    var daysOfWeek = ['일', '월', '화', '수', '목', '금', '토'];
    for (var i = 0; i < daysOfWeek.length; i++) {
      calendarHtml += '<th>' + daysOfWeek[i] + '</th>';
    }
    calendarHtml += '</tr><tr>';
  
    var firstDay = new Date(year, month - 1, 1).getDay();
    var lastDate = new Date(year, month, 0).getDate();
  
    for (var i = 0; i < firstDay; i++) {
      calendarHtml += '<td></td>';
    }
  
    for (var i = 1; i <= lastDate; i++) {
      calendarHtml += '<td>' + i + '</td>';
      if ((i + firstDay) % 7 === 0 && i !== lastDate) {
        calendarHtml += '</tr><tr>';
      }
    }
  
    calendarHtml += '</tr></table>';
  });
  

  async function postHandler() {
    console.log('postHandler 함수 호출'); // 함수가 호출되는지 확인
  
    const title = document.getElementById('diarytitle').value;
    const content = document.getElementById('diaryContent').value;
    const post_date = document.getElementById('diaryDate').value;
    const { userId } = window.config;
  
    try {
      const response = await fetch(`/post/write/${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ title, content, post_date })
      });
  
      const result = await response.json();
  
      if (response.ok) {
        alert(result.message);
        window.location.href = `/post/${userId}`;
      } else {
        console.error('서버 응답 오류:', result);
        alert('게시물 작성 중 오류가 발생했습니다.');
      }
    } catch (error) {
      console.error('fetch 요청 중 오류:', error);
      alert('서버와의 통신 중 오류가 발생했습니다.');
    }
  }



  function deleteDiary(postId) {
    // AJAX를 사용하여 서버로 삭제 요청을 보냅니다.
    fetch(`/post/delete/${postId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('다이어리가 성공적으로 삭제되었습니다.');
            // 성공적으로 삭제되면 페이지를 새로고침하여 업데이트된 다이어리 리스트를 보여줍니다.
            location.reload();
        } else {
            alert('다이어리 삭제 중 오류가 발생했습니다.');
        }
    })
    .catch(error => {
        console.error('다이어리 삭제 중 오류:', error);
        alert('다이어리 삭제 중 오류가 발생했습니다.');
    });
}


document.getElementById('diaryForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // 기본 폼 제출 방지
    await postHandler(); // postHandler 함수 호출
});

