package com.web.p6;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import jakarta.servlet.http.HttpSession;

@Controller
public class WorldCup {

	@GetMapping("/star/1")
	public String star1() {
		return "star1";
	}
	
	@GetMapping("/star/2")
	public String star2(HttpSession se, @RequestParam("choice1") String choice1) {
		se.setAttribute("choice1", choice1);
		return "star2";
	}
	
	@GetMapping("/star/3")
	public String star3(HttpSession se, @RequestParam("choice2") String choice2, Model mo) {
		se.setAttribute("choice2", choice2);
		mo.addAttribute("choice1", se.getAttribute("choice1"));
		mo.addAttribute("choice2", se.getAttribute("choice2"));
		return "star3";
	}
	
	@GetMapping("/star/winner")
	public String starWinner(@RequestParam("winner") String winner, Model mo) {
		mo.addAttribute("winner", winner);
		return "starWinner";
	}
}
