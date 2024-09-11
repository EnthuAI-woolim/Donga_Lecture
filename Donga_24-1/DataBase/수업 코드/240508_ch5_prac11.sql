desc usertable;
select * from usertable;
use shop_db;
# Q.1-1
alter table usertable modify regyear smallint;

# Q.1-2
alter table usertable change column email mail char(20);
alter table usertable change column mail email char(20);

# Q.1-3
alter table usertable add phone char(20);
alter table usertable drop column phone;

# Q.1-4
alter table usertable rename users; -- 1
rename table users to usertable; 	-- 2

# Q.1-5
rename table usertable to market_db.usertable;
rename table market_db.usertable to shop_db.usertable;

# Q.2
create database naver_db;
use naver_db;

# Q.3
create table member (
	mem_id 		char(8) 	not null primary key,
    mem_name	varchar(10)	not null,
    mem_number	tinyint		not null,
    addr		char(2)		not null,
    phone1		char(3),
    phone2		char(8),
    height		tinyint 	unsigned,
    debut_date	date
);

# Q.4
insert into member values('TWC', '트와이스', 9, '서울', '02', '11111111', 167, '2015.10.19'); 

# Q.5
create table buy(
	num		int 	not null auto_increment,
    mem_id 	char(8)	not null,
    prod_name	char(6)	not null,
    group_name 	char(4),
    price	int 	unsigned 	not null ,
    amount	smallint	unsigned	not null,
    primary key(num), foreign key(mem_id) references member(mem_id)
);

select * from member;
select * from buy;

# Q.6
insert into buy values(0, 'BLK', '지갑', null, 30, 2);












