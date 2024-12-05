package com.web.p6;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;

@Entity
public class star {
	@Id
	public Integer huno;
	public String name;
	public Integer fcount;
}
