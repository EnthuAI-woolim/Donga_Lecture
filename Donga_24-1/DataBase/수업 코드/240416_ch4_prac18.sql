# Q.13
# 1. select(cace)의 경우 : end뒤에 아무것도 안적어줌. alias를 적어줄 순 있음
select M.mem_id as 아이디, M.mem_name as 이름, sum(price*amount) as 총구매액,
	case
		when sum(price*amount) then '최우수고객'
        when sum(price*amount) then '우수고객'
		when sum(price*amount) then '그린고객'
        else '잠재고객'
	end as 회원등급
from buy B
	right outer join member M
	on M.mem_id = B.mem_id
group by M.mem_id
order by 총구매액 desc;

# Q.14
select * from member;
select left(mem_id, 1) id, repeat(right(mem_name, 1), mem_number) name, reverse(addr) addr from member;

# Q.15-1
CREATE TABLE buy2 (SELECT mem_id, price FROM buy);

# Q.15-2
PREPARE myQuery FROM 'update buy2 set price = price * ?';

# Q.15-3
SET @rate = 1.1;
EXECUTE myQuery USING @rate;

# Q.15-4
SET @rate = 0.9;
EXECUTE myQuery USING @rate;

# Q.15-5
DEALLOCATE PREPARE myQuery;

# Q.16
use world;
SELECT UPPER(name) '도시', FORMAT(population, 0) '인구수' FROM city
WHERE countryCode = 'KOR'
ORDER BY population DESC
LIMIT 5;
# LIMIT 0, 5;
# LIMIT 5 OFFSET 0; OFFSET 뒤에는 시작 인덱스를 적어줌

# Q.17
# 1.
SELECT co.name '나라',  COUNT(ci.name) '도시수'
FROM city AS ci
INNER JOIN country AS co
ON ci.countrycode = co.code
GROUP BY 나라
ORDER BY 도시수 DESC
LIMIT 10;

# 2.
SELECT co.name '나라',  COUNT(ci.name) '도시수'
FROM city AS ci, country AS co
WHERE ci.countrycode = co.code
GROUP BY 나라
ORDER BY 도시수 DESC
LIMIT 10;

# Q.18
# (1)
SELECT code '국가코드', name '국가명'
FROM country
WHERE code not in (select distinct countrycode from city);

# (2)
# 'country테이블에는 있지만'이니깐 country테이블을 기준으로 code가 같은 것을 가져와야
# city테이블에서 countrycode가 없는 행에는 null이 들어감
SELECT country.code 국가코드, country.name 국가명
from country
left outer join city
on country.code = city.countrycode
where city.countrycode is null;
