# p.7
use market_db;
create table table1 (
	col1 int primary key,
    col2 int,
    col3 int
);
show index from table1;

create table table2 (
	col1 int primary key,
    col2 int unique,
    col3 int unique
);
show index from table2;

# p.11~
use naver_db;
drop table if exists buy, member;
create table member (
	mem_id char(8),
    mem_name varchar(10),
    mem_number int,
    addr char(2)
);
select * from member;

insert into member values('TWC', '트와이스', 9, '서울');
insert into member values('BLK', '블랙핑크', 4, '경남');
insert into member values('WMN', '여자친구', 6, '경기');
insert into member values('OMY', '오마이걸', 7, '서울');

alter table member
	add constraint
    primary key (mem_id);

alter table member drop primary key;
alter table member
	add constraint
    primary key(mem_name);
select * from member;

insert into member values('GRL', '소녀시대', 8, '서울');

# p.16
drop table if exists member;
select * from member;
show index from member;

create table member (
	mem_id char(8),
    mem_name varchar(10),
    mem_number int,
    addr char(2)
);

insert into member values('TWC', '트와이스', 9, '서울');
insert into member values('BLK', '블랙핑크', 4, '경남');
insert into member values('WMN', '여자친구', 6, '경기');
insert into member values('OMY', '오마이걸', 7, '서울');

# p.17
alter table member
	add constraint
	unique (mem_id);

# p.18
alter table member
	add constraint
	unique (mem_name);

# 처음 만들때, unique와 not null하면 해당 컬럼 기준으로 정렬됨
# 하지만 이럴바에 그냥 primary key를 쓰는 경우가 대부분이다.
drop table if exists member;
select * from member;
show index from member;

create table member (
	mem_id char(8),
    mem_name varchar(10) unique not null,
    mem_number int,
    addr char(2)
);

insert into member values('TWC', '트와이스', 9, '서울');
insert into member values('BLK', '블랙핑크', 4, '경남');
insert into member values('WMN', '여자친구', 6, '경기');
insert into member values('OMY', '오마이걸', 7, '서울');

insert into member values('GRL', '소녀시대', 8, '서울');