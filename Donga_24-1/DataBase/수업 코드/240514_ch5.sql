# ch5-2
# ch) p.29
# DDL - create, alter, drop
# create 뒤에 개체가 나옴(database, table, index)

# 5-3. view
create view v_member as 
select mem_id, mem_name, addr from member;

select * from v_member;
drop view v_member;

desc v_member;

# view 생성코드 보기
show create view v_member;

# p.32
# 뷰를 update해서 원본 테이블의 데이터 변경 - 성공
update v_member set addr = '부산' where mem_id = 'BLK';
select * from member;

# 뷰를 insert해서 원본 테이블에 데이터 삽입 - 실패
insert into v_member values('BTS', '방탄소년단', '경기');
# 원본 테이블의 not null 컬럼이 뷰에는 없어서, insert하면 오류남

# not null컬럼인 mem_number를 null로 수정하고, 위의 코드 실행함.
alter table member modify mem_number int null;
desc member;

check table v_member;
