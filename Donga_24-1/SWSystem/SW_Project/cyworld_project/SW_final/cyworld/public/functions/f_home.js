document.addEventListener('DOMContentLoaded', function() {
    const { userId } = window.config;
   
     const uploadButton = document.getElementById('miniroom-edit');
     const imageUpload = document.getElementById('miniroom-upload');
     const imageContainer = document.getElementById('image-container');
     const miniroomText = document.getElementById('miniroom-text');
     console.log(userId);  // 확인용
   
     uploadButton.addEventListener('click', function() {
       imageUpload.click();
     });
   
     imageUpload.addEventListener('change', async function(event) {
       const file = event.target.files[0];
 
       if (file) {
         const formData = new FormData();
         formData.append('miniroomImage', file);
         formData.append('id', userId);  // 사용자 ID 전송
 
         // console.log(file);
   
         try {
           const response = await fetch(`/home/edit_miniroom/${userId}`, {
             method: 'PATCH',
             body: formData
           });
           
          
           const result = await response.json();
           if (result.success) {
             miniroomText.innerHTML = '';
             alert(result.message);    
 
             // 이미지를 화면에 표시하는 코드는 여기에서 추가
             // 이미지의 URL은 result.imageUrl에서 가져와서 사용
             const img = document.createElement('img');
             img.src = result.imageUrl;
             img.classList.add('miniroom-box-img'); // 이미지에 클래스 추가 (선택 사항)
             imageContainer.innerHTML = ''; // 이미지 컨테이너 내용 비우기
 
             imageContainer.appendChild(img);
           } else {
             alert('이미지 업로드 실패: ' + result.message);
           }
         } catch (err) {
           console.error('이미지 업로드 중 오류:', err);
           alert('이미지 업로드 중 오류가 발생했습니다.');
         }
       }
     });
   });
   