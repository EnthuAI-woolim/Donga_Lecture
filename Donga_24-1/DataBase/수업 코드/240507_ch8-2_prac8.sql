# Q.1
# 추가
alter table usertable add primary key(userid);
# DML에는 table키워드 없는데 데이블을 다루는 DDL에는 table키워드 있다.

# 제거(기본키)
alter table usertable drop primary key;

# 수정(기본키로 설정하면 자동으로 not null이 되어있는 상태임)
alter table usertable modify userid char(15) default null;

# Q.2
insert into usertable (userid, username) values('dooli', '둘리');
insert into usertable values('maru', '마루', null, 2023);
insert into usertable values('BLK', '블랙핑크', '1111@email.com', null);


desc usertable;
select * from usertable;