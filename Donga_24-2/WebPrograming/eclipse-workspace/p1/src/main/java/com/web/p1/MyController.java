package com.web.p1;

import java.util.ArrayList;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

// 엄청나게 중요하게 신경써야 하는 부분
// 컨트롤러 선언(@Controller). 이걸 해줘야 고객응대 가능!!
@Controller
public class MyController { // 소스명, 클래스명 대소문자까지 완벽 동일
	
	// 여기 코딩할 건데
	// 이 안에는 메소드들을 코딩합니다. 화면1개에 메소드1개
	
	// # 수업코드
	@GetMapping("/")	// 고객님이 사이트주소만 넣고 더이상 세부주소 안치시면 (홈페이지 첫화면)
	public String home() {	// 메소드명 자유. 대부분은 화면 이름이나 주소랑 동일하게 하는 경향
		return "home";	// 화면 내보내는 한 줄. html소스는 templates안에 있어야 함
	}					// thymeleaf가 templates 폴더에서 확장자가 html인 home을 선택
	
	@GetMapping("/ex01")
	public String ex01() {
		return "ex01"; 
	}
	
	@PostMapping("/ex01result")
	public String ex01Answer(@RequestParam(name="mid") String mid, @RequestParam(name="pw") String pw, Model mo) {
		mo.addAttribute("mid", mid);
		mo.addAttribute("pw", pw);
		return "ex01Answer";
	} 
	
	@GetMapping("/ex02")
	public String ex02() {
		return "ex02";
	}
	
	@GetMapping("/ex02/answer")
	public String ex02Answer(@RequestParam("mname") String mname, @RequestParam("po") String po, Model mo) {
		mo.addAttribute("mname", mname);
		mo.addAttribute("po", po);
		int salary = 0;
		switch(po){ // java switch문에서는 break없음 -> 한줄 하고 ;만나서 종료함
			case "사원" -> salary = 3500; /* Java switch문 */
			case "대리" -> salary = 5000; 
			case "팀장" -> salary = 7000; 
			case "임원" -> salary = 9900; 
		}
		mo.addAttribute("salary", salary);
		return "ex02Answer";
	}
	
	@GetMapping("/ex03")
	public String ex03() {
		return "ex03";
	}
	
	@PostMapping("/ex03/answer")
	public String ex03Answer(@RequestParam("mname") String n, @RequestParam("color") String c, Model mo) {
		mo.addAttribute("n", n);
		mo.addAttribute("c", c);
		return "ex03Answer";
	}
	
	@GetMapping("/ex04")
	public String ex04(Model mo) {
		// var: 자동 타입 설정
		var arr = new ArrayList<String>();
		arr.add("고흐"); 
		arr.add("james");
		arr.add("dooli");
		arr.add("bread"); 
		/* 지금은 회원정보 하드코딩. 나중에는 database에서 가져옴 */
		mo.addAttribute("arr", arr);
		return "ex04";
	}
	
	@GetMapping("/login")
	public String login() {
		return "login";
	}

	
	// # 과제
	@GetMapping("/assignment")
	public String assignment() {
		return "assignment";
	}
	
	// ## Ch3
	// ### quiz 03
	@GetMapping("/assignment/ch3/wise")
	public String wise() {
		return "wise"; 
	}
	
	@PostMapping("/assignment/ch3/wise/answer")
	public String wiseAnswer(@RequestParam(name="mname") String mname, @RequestParam(name="content") String content, Model mo) {
		mo.addAttribute("mname", mname);
		mo.addAttribute("content", content);
		return "wiseAnswer";
	}
	
	// ### quiz 05
	@GetMapping("/assignment/ch3/bread")
	public String bread() {
		return "bread";
	}
	
	@PostMapping("/assignment/ch3/bread/answer")
	public String breadAnswer(
			@RequestParam("name") String name,
			@RequestParam("price") Integer price,
			@RequestParam("num") Integer num,
			Model mo
			) {
		mo.addAttribute("name", name);
		mo.addAttribute("price", price);
		mo.addAttribute("num", num);
		// 숫자는 int와 Integer 중 Integer권장
		// Integer는 래퍼클래스라서 객체타입이라 null값을 받을 수 있음
		return "breadAnswer";
	}
	
	// ### quiz 06
	@GetMapping("/assignment/ch3/q06")
	public String ch3_q06() {
		return "ch3_q06";
	}

	@GetMapping("/assignment/ch3/q06/a")
	public String ch3_q06a() {
		return "ch3_q06a";
	}
	
	@PostMapping("/assignment/ch3/q06/aa")
	public String ch3_q06aa(@RequestParam("first") String first, @RequestParam("second") String second, Model mo) {
		mo.addAttribute("first", first);
		mo.addAttribute("second", second);
		return "ch3_q06aa";
	}
	
	@GetMapping("/assignment/ch3/q06/b")
	public String ch3_q06b() {
		return "ch3_q06b";
	}
	
	@PostMapping("/assignment/ch3/q06/bb")
	public String ch3_q06bb(@RequestParam("field") String field , Model mo) {
		mo.addAttribute("field", field);
		return "ch3_q06bb";
	}
	
}
