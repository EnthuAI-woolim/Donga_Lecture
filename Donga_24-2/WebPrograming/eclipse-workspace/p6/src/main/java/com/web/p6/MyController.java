package com.web.p6;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;

@Controller
public class MyController {

	@GetMapping("/timeTable")
	public String timeTable() {
		return "timeTable";
	}

}
