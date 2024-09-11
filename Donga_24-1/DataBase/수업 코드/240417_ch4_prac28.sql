# Q.19
use market_db;
create table pay2 (select customer_id, amount, payment_date from sakila.payment limit 5000);
select * from pay2;

# Q.20
desc pay2;

# Q.21
select min(payment_date) 최소, max(payment_date) 최대 from pay2;

# Q.22
insert into pay2 values(100, 10.23, now());
select * from pay2 order by payment_date desc;

insert into pay2 values(1, 1.23, '2024-05-01 10:23:45');
insert into pay2 values(1, 1.23, '2024-05-02');
insert into pay2 values(1, 1.23, '2024-5-3'); 
insert into pay2 values(1, 1.23, '2024.5.4 10-23-45');
insert into pay2 values(1, 1.23, '10:23:45');
insert into pay2 values(1, 1.23, '10:10:10');

# Q.23
# (1)
delete from pay2 where payment_date < '2005-01-01' or payment_date > '2006-12-31';
# (2)
delete from pay2 where payment_date not between '2005-01-01 00:00:00' and '2006-12-31 23:59:59';

# Q.24
select * from pay2;

select left(payment_date, 7) 년월, count(*) 건수, sum(amount) 합 from pay2
group by 년월 with rollup;


# Q.25
select P.customer_id 아이디, concat(C.first_name, ' ', C.last_name), count(P.customer_id) 건수, sum(P.amount) 합
from pay2 P
inner join sakila.customer C
on P.customer_id = C.customer_id
group by P.customer_id
order by 아이디 
limit 5;

select p.customer_id 아이디, concat(c.first_name, ' ', c.last_name) 이름, count(*) 건수, sum(p.amount) 합
from pay2 p, sakila.customer c
where p.customer_id = c.customer_id
group by p.customer_id
limit 5;

# Q.26
use sakila;

select first_name, count(*) 수 
from customer
group by first_name
having  수 >= 2
order by first_name;

# Q.27 - 난이도 상
# (1)
select concat(first_name, ' ', last_name) 동명이인
from customer
where first_name in (select first_name from customer group by first_name having  count(*) >= 2)
order by 동명이인;

# (2)-1
select concat(A.first_name, ' ', A.last_name) 동명이인
from customer A
	inner join customer B
	on A.first_name = B.first_name
	and A.last_name != B.last_name
order by 동명이인;

# (2)-2
select concat(a.first_name, ' ', a.last_name) 동명이인
from customer a, customer b
where a.first_name = b.first_name
	and a.last_name != b.last_name
order by 동명이인;

# Q.28
# 파이썬
-- import pymysql

-- conn = pymysql.connect(host='127.0.0.1', user='root', password='root', db='market_db', charset='utf8')
-- cur = conn.cursor()

-- sql = "drop table if exists pay2"
-- cur.execute(sql)
-- conn.commit()
-- conn.close()
