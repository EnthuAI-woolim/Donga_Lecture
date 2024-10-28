package com.web.p1;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import jakarta.servlet.http.HttpSession;

@Controller
public class ShoppingController {
	
	@GetMapping("/login")
	public String login(HttpSession se, Model mo) {
		// (String) : 지역변수에 할당하게하기위해 형변환
		String mid = (String)se.getAttribute("mid"); 
		mo.addAttribute("loginInfo", mid==null?"로그인하세요":mid+"님");
		return "login";
	}


}
