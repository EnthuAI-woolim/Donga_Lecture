package com.web.p4;

import java.util.ArrayList;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

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

	
	@GetMapping("/login/check")
	public String loginCheck(HttpSession se, @RequestParam("mid") String mid, @RequestParam("pw") String pw) {
		var arr = new ArrayList<Member>(); 
		arr.add(new Member("고흐","g")); arr.add(new Member("james","j"));
		arr.add(new Member("dooli","d")); arr.add(new Member("iu","i"));
		
		boolean ourMember=false;
		for(var a: arr) {
			if(a.mid.equals(mid) && a.pw.equals(pw)) { 
				ourMember = true; 
				break; 
			}
		} 
		
		if( ourMember ) { 
			se.setAttribute("mid", mid); 
			return "redirect:/menu";  // redirect: -> 'html화면이 아니라, url이 "/menu"인 메소드를 실행시켜라'라는 뜻
		} else { 
			se.setAttribute("msg", mid + "는 미등록 아이디이거나 혹은 패스워드가 일치하지 않습니다.\n"
					+ "확인 후 로그인 부탁드립니다.");
			return "redirect:/popup"; 
		} 
	}
	
	@GetMapping("/logout")
	public String logout(HttpSession se, Model mo) {
		mo.addAttribute("mid", se.getAttribute("mid"));
		se.invalidate(); 
		return "logout";
	}
	
	@GetMapping("/menu")
	public String menu(HttpSession se, Model mo) {
		mo.addAttribute("mid", se.getAttribute("mid"));
		return "menu";
	}
	
	@GetMapping("/popup")
	public String popup(HttpSession se, Model mo) { 
		mo.addAttribute("msg", se.getAttribute("msg")); 
		se.removeAttribute("msg");
		return "popup";
	}
	
	@GetMapping("/coffee")
	public String coffee(HttpSession se, Model mo) {
		mo.addAttribute("mid", se.getAttribute("mid"));
		return "coffee";
	}
	

}
