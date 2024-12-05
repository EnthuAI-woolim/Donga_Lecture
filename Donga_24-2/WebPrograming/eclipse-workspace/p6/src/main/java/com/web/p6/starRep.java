package com.web.p6;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.transaction.annotation.Transactional;

//import jakarta.transaction.Transactional;

public interface starRep extends JpaRepository<star, Integer> {
	@Transactional
	@Modifying
	@Query("update star set fcount = fcount+1 where name = ?1")
	void increaseFcount(String name);
	
	@Transactional
	@Modifying
	@Query("update star set fcount = 0")
	void starReset();
}
