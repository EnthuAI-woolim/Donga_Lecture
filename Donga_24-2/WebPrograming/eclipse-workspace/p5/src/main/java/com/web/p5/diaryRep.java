package com.web.p5;

import org.springframework.data.jpa.repository.JpaRepository;

public interface diaryRep extends JpaRepository<diary, Integer> {} 

// 소스 만들때 New-Class X -> New-Interface
// JpaRepository<테이블명, 첫컬럼 데이터 타입>