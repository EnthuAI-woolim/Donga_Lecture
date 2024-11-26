# Quiz 12
create table product (
order_id int not null primary key, 
customer_id varchar(10) not null,
product_name varchar(30) not null,
quantity int not null,
rdate datetime not null
);

insert into product values
(1, 'aaa', 'iPad Pro', 2, now()),
(2, 'bbb', 'iPad Air', 3, now()),
(3, 'ccc', 'iPad', 1, now()),
(4, 'ddd', 'iPad mini', 4, now());

select * from product;

# Quiz 16
drop table if exists diary;

create table diary (
	no int auto_increment primary key,
    je varchar(30),
    nae varchar(100),
    wdate datetime
);

insert into diary values (null, 'test', 'Hello, world!', now());
select * from diary;

# Quiz 21
drop table if exists survey;
create table survey (
	no int auto_increment primary key,
	userid varchar(10) not null,
	food varchar(30) not null,
	dessert varchar(30) not null,
	sdate datetime not null
);



