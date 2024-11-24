package com.web.p5;

import java.time.LocalDateTime;
import org.hibernate.annotations.CreationTimestamp;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;

@Entity
public class product {
	
	@Id public Integer order_id;
	public String customer_id;
	public String product_name;
	public Integer quantity;
	@CreationTimestamp public LocalDateTime rdate;
	
} // class