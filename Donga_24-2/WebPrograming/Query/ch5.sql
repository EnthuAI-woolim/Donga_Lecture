create database airdb;
use airdb;
show tables;

create table member (
mid varchar(10) not null primary key, 
pw varchar(10) not null,
name varchar(30) not null,
phone varchar(15) not null,
mileage int not null, 
rdate datetime not null
);

DESC member;

DROP TABLE member; 

insert into member values
 ('ann','1111','빨강머리 앤', '010-999-9999',0, now()),
 ('dooli','5252','둘리 사우르스', '곧 구입 예정',0, now()),
 ('james','7777','제임스 본드', '007-007-7777',1000000, now());
 
update member
set phone='010-111-2222', pw='8888' 
where id='ann';

delete from member where mid='ann';

delete from member;

select * from member;


