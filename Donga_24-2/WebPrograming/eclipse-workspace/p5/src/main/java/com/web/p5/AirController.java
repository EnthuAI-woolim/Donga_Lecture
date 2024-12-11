package com.web.p5;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class AirController {
	
	 @Autowired
	 private memberRep mrep;
	 @Autowired
	 private productRep prep;
	 @Autowired
	 private diaryRep drep;
	 
	 // ch5_p.21
	 @GetMapping("/memberList")
	 public String memberList(Model mo) {
		 List<member> arr = mrep.findAll(); 
		 mo.addAttribute("arr",arr);
		 return "memberList";
	 }
	 
	// Quiz 12
	 @GetMapping("/productList")
	 public String productList(Model mo) {
		 List<product> arr = prep.findAll();
		 mo.addAttribute("arr", arr);
		 return "productList";
	 }
	 
	 // Quiz 13 ~ 20
	 @GetMapping("/diary")
	 public String diary() { 
		 return "diary";
	 }
	 
	 @GetMapping("/diary/insert")
	 public String diaryInsert(@RequestParam("je") String je, @RequestParam("nae") String nae) {
		 diary d = new diary();
		 d.je = je;
		 d.nae = nae;
		 drep.save(d); // insert
		 return "redirect:/diaryPop";
	 }
	 
	 @GetMapping("/diaryPop")
	 public String diaryPop() { 
		 return "diaryPop";
	 }
	 
	 @GetMapping("/diaryList")
	 public String diaryList(Model mo) {
		 mo.addAttribute("arr", drep.findAll());
		 return "diaryList";
	 }
	 
	 
	 
	 
	 
} 