package com.web.p5;

import org.springframework.data.jpa.repository.JpaRepository;

public interface memberRep extends JpaRepository<member, String> {
	
} 

// 소스 만들때 New-Class X -> New-Interface