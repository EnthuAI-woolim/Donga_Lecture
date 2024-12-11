package com.web.p5;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import jakarta.servlet.http.HttpSession;

@Controller
public class SurveyController {
	
	@Autowired
	private surveyRep srep;
	
	// Quiz 21 ~ 26
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
		se.setAttribute("a2", a2);
		
		survey s = new survey();
		s.userid = (String) se.getAttribute("mid");
		s.food = (String) se.getAttribute("a1");
		s.dessert = (String) se.getAttribute("a2");
		srep.save(s);
		
		return "thanks";
	}
	
	@GetMapping("/result")
	public String result(HttpSession se, Model mo) {
		mo.addAttribute("mid", se.getAttribute("mid"));
		mo.addAttribute("a1", se.getAttribute("a1"));
		mo.addAttribute("a2", se.getAttribute("a2"));
		
		return "result";
	}
	
	@GetMapping("/surveyList")
	public String surveyList(Model mo) {
		mo.addAttribute("arr", srep.findAll());
		return "surveyList";
	}

}
