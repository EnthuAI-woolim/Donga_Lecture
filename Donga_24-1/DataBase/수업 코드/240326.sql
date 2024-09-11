CREATE TABLE city_popul (
	city_name CHAR(35),
	populaion INT
);

INSERT INTO city_popul SELECT Name, Population FROM world.city;

SELECT * FROM city_popul;

DROP TABLE IF EXISTS city_popul;

CREATE TABLE city_popul (
	SELECT name as city_name, population as city_population
    FROM world.city
);

CREATE TABLE big_table1 (SELECT * FROM world.city, sakila.country);
CREATE TABLE big_table2 (SELECT * FROM world.city, sakila.country);
CREATE TABLE big_table3 (SELECT * FROM world.city, sakila.country);

SELECT * FROM big_table2;
SELECT * FROM world.city;
SELECT * FROM sakila.country;

DROP TABLE big_table2;
DELETE FROM big_table2;
TRUNCATE TABLE big_table3;



