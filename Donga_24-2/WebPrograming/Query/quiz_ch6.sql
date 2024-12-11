#ch6 Quiz

# 5
drop table if exists star;
create table star (
	huno 	int 		not null primary key,
    name 	varchar(20)	not null,
    fcount 	int 		not null
);

insert into star values(1, '람보르기니', 0);
insert into star values(2, '옵티머스', 0);
insert into star values(3, '해바라기', 0);
insert into star values(4, '둘리', 0);

select * from star;

# 6
update star set fcount = fcount+1 where name = '해바라기';

# 7
update star set fcount = 0;

select * from member;