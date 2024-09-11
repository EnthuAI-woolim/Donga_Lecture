SELECT * FROM market_db.member;

# Q.2
SELECT mem_name, height
FROM member
WHERE height = (SELECT max(height) FROM member)
	OR height = (SELECT min(height) FROM member);
    
# Q.3
# (1)
SELECT addr, count(*)
FROM member
GROUP BY addr WITH ROLLUP;

# (2)
SELECT addr, mem_name, count(*)
FROM member
GROUP BY addr, mem_name WITH ROLLUP;

# Q.8
# (1)
select ci.name 도시이름, co.name 나라이름, ci.population 도시인구 
from city ci, country co
where ci.countryCode = co.code
	and ci.population >= 9000000
order by ci.population desc;

# (2)
select ci.name 도시이름, co.name 나라이름, ci.population 도시인구 
from city ci
	inner join country co
    on ci.countryCode = co.code
where ci.population >= 9000000
order by ci.population desc;

# Q.9
# (1)
select co.name 나라이름, co.code, count(*) 공식언어수
from country co, countrylanguage cl
where co.code = cl.countrycode and cl.isofficial = 'T'
group by co.code
having 공식언어수 >= 3
order by 공식언어수 desc;

# (2)
select co.name 나라이름, co.code, count(*) 공식언어수
from country co
	inner join countrylanguage cl
	on co.code = cl.countrycode and cl.isofficial = 'T'
group by co.code
having 공식언어수 >= 3
order by 공식언어수 desc;