package com.web.p6;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class MemberController {
	
	@Autowired
	private memberRep mrep;
	
	// Quiz 14 ~ 17
	@GetMapping("/member/new")
	public String memberNew() {
		return "memberNew";
	}
	
	@GetMapping("/member/insert")
	public String memberInsert(
			@RequestParam("mid") String mid, 
			@RequestParam("pw") String pw, 
			@RequestParam("name") String name, 
			@RequestParam("phone") String phone, Model mo) {
		if( mrep.existsById(mid) ) {
			mo.addAttribute("msg", mid+"는 이미 사용되고 있는 아이디입니다.");
			mo.addAttribute("url", "back");
		} else {
			member m = new member();
			m.mid = mid; m.pw = pw;
			m.name = name; m.phone = phone;
			m.mileage = 1000;
			mrep.save(m);
			mo.addAttribute("msg", mid+"님, 반갑습니다!! (회원 리스트로 이동)");
			mo.addAttribute("url", "/member/list"); 
		 }
		return "popup";
	}
	
	@GetMapping("/member/list")
	public String memberList(Model mo) {
		mo.addAttribute("arr", mrep.findAll());
		return "memberList";
	}

}
