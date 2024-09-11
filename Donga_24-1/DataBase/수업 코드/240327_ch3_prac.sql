# ch3_prac - Q.1
USE world;

# ch3_prac - Q.2
SHOW TABLES;

# ch3_prac - Q.3
SELECT count(*) FROM country;

# ch3_prac - Q.4
SELECT * FROM country WHERE name LIKE '%ko%';

# ch3_prac - Q.5
SELECT * FROM city WHERE countrycode = (SELECT code FROM country WHERE name='South Korea');

# ch3_prac - Q.6-1
SELECT * FROM countrylanguage WHERE language='Korean';
# ch3_prac - Q.6-2
SELECT * FROM countrylanguage WHERE countrycode = 'USA';
# ch3_prac - Q.6-3
SELECT language, count(*) AS 나라수 FROM countrylanguage 
GROUP BY language
ORDER BY 나라수 DESC
LIMIT 5;
