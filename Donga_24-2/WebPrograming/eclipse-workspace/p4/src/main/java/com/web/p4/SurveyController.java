package com.web.p4;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import jakarta.servlet.http.HttpSession;

@Controller
public class SurveyController {
	
	@GetMapping("/start")
	public String start() {
		return "start";
	}
	
	@PostMapping("/survey1")
	public String survey1(HttpSession se, @RequestParam("mid") String mid) {
		se.setAttribute("mid", mid);
		return "survey1";
	}
	
	@PostMapping("/survey2")
	public String musician(HttpSession se, @RequestParam("a1") String a1) {
		se.setAttribute("a1", a1);
		return "survey2";
	}
	
	@PostMapping("/thanks")
	public String thanks(HttpSession se, @RequestParam("a2") String a2, Model mo) {
		
		if (a2.equals("아이스크림") || a2.equals("빵")) {
			a2 = a2 + "을";
		} else {
			a2 = a2 + "를";
		}
		se.setAttribute("a2", a2);
		return "thanks";
	}
	
	@GetMapping("/result")
	public String result(HttpSession se, Model mo) {
		mo.addAttribute("mid", se.getAttribute("mid"));
		mo.addAttribute("a1", se.getAttribute("a1"));
		mo.addAttribute("a2", se.getAttribute("a2"));
		return "result";
	}

}
