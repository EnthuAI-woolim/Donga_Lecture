# Q.13
# 테이블 타입까지 다 볼 수 있다.
# 행(row)가 실제와 다른 경우가 많다.
select * from information_schema.tables where table_schema='market_db';

use market_db;
show tables;

# 실제 행(row) 수
select count(*) from buy;

# Q.14
# 뷰를 통해 업데이트되는지 확인가능 (IS_UPDATABLE)
select * from information_schema.views where table_schema='market_db';

# Q.15
# 현재 use market_db; 되어 있는 상태
show tables from world;

# Q.16
# 테이블 타입까지 보기, 다른 데이터베이스의 데이블 타입까지 보기;
show full tables;
show full tables from sakila;

# Q.17
# 1.
# PK는 하나니까 그냥 drop primary key 하면 된다. PK는 여러개 가능하므로 어느 PK인지 지정해줘야 된다.
alter table buy drop foreign key buy_ibfk_1;

# 2.
alter table buy add foreign key(mem_id) references member(mem_id) on update cascade;

# 3.
update member set mem_id = 'PINK' where mem_id = 'BLk';

select * from market_db.member;
select * from market_db.buy;

# 테이블 우클릭 - send to SQL Edior - Create statement
CREATE TABLE `buy` (
  `num` int NOT NULL AUTO_INCREMENT,
  `mem_id` char(8) NOT NULL,
  `prod_name` char(6) NOT NULL,
  `group_name` char(4) DEFAULT NULL,
  `price` int NOT NULL,
  `amount` smallint NOT NULL,
  PRIMARY KEY (`num`),
  KEY `mem_id` (`mem_id`),
  CONSTRAINT `buy_ibfk_1` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

# 원상복구
update member set mem_id = 'BLK' where mem_id = 'PINK';

# Q.18
create or replace view v_height
as
select * from member where height >= 167;

insert into v_height values ('idle', '아이들', 5, '하단', '051', '55556666', 159, now());

# Q.19
create or replace view v_height
as
select * from member where height >= 167 with check option;

insert into v_height values ('dooli', '둘리', 1, '하단', '051', '55556666', 140, now());
insert into v_height values ('dooli', '둘리', 1, '하단', '051', '55556666', 177, now());

SELECT * FROM market_db.v_height;

# Q.20
use naver_db;
# primary key가 다름
# (1) : 데이터의 타입까지 모두 복사해서 가져오기
create table city (
	id int auto_increment not null,
    name char(35) not null,
    primary key(id)
);
insert into city (select id, name from world.city);
desc city;

# (2) : 데이터만 가져오기
create table city1 (select id, name from world.city);
desc city1;

# Q.21
use naver_db;
# 압축하는데 시간이 걸리더라도 데이터 양이 너무 많다면
drop table if exists city1, citycom;
create table city (id int, name char(35));
create table citycom (id int, name char(35)) row_format=COMPRESSED;

insert into city select id, name from world.city;
insert into citycom select id, name from world.city;

# Q.22
# row_format 행에서 테이블의 상태 확인가능
show table status from naver_db;

