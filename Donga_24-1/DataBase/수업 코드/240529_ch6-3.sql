show index from member;

# 특정 테이블만 보고싶으면 like 사용('member%' : member라는 글자로 시작하는 테이블)
# 작은 따옴표 주의!!
show table status like 'member';
show table status like 'member%';
show table status from market_db;

# 인덱스 create, 중복 허용
create index idx_member_addr on member(addr); 
create index idx_member_mem_number on member(mem_number);

# 중복 허용 X
create unique index idx_member_addr on member(addr); 

analyze table member;
show table status like 'member';

# p.49
# 고유 인덱스 생성하기
create unique index idx_member_mem_name on member(mem_name);
ALTER TABLE member ADD UNIQUE (mem_name);

DROP INDEX idx_member_mem_name ON member;
ALTER TABLE member DROP INDEX idx_member_mem_name;

show index from member;

select * from member;
desc member;

# 인덱스 활용 실습
select * from member where mem_name = '소녀시대';


# 자신을 참조하는 데이블들 확인 후 FK, 인덱스, PK순으로 제거
# select * from 해도 됨
# FK를 확인하는 SELECT문
select table_name, constraint_name
from information_schema.referential_constraints
where constraint_schema='market_db' and referenced_table_name='member';

# member테이블의 PK를 참고하고 있는 buy테이블의 FK를 DROP
alter table buy drop foreign key buy_ibfk_1;

# member테이블의 고유인덱스 전부 DROP
DROP INDEX idx_member_mem_name ON member;
DROP INDEX idx_member_mem_number ON member;
DROP INDEX idx_member_addr ON member;

# PK DROP
alter table member drop primary key;

show index from member;
desc member;



