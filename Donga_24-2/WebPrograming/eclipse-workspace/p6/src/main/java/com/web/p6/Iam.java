package com.web.p6;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;


@Controller
public class Iam {
	
	@GetMapping("/iam")
	public String iam() {
		return "iam";
	}
	
	@PostMapping("/iamAnswer")
	public String iamAnswer(
			@RequestParam("gender") String gender,
			@RequestParam("age") String age,
			@RequestParam(name="foods", required=false) String[] foods,
			@RequestParam("face") String face,
			@RequestParam("grade") String grade,
			@RequestParam("attention") String attention,
			@RequestParam("comments") String comments,
			Model mo
			) {
		
		if (foods == null || foods.length == 0) {
			foods = new String[] { "선택없음" };
	    }
		
		mo.addAttribute("gender", gender);
		mo.addAttribute("age", age);
		mo.addAttribute("foods", foods);
		mo.addAttribute("face", face);
		mo.addAttribute("grade", grade);
		mo.addAttribute("attention", attention);
		mo.addAttribute("comments", comments);
		return "iamAnswer";
	}
	
}