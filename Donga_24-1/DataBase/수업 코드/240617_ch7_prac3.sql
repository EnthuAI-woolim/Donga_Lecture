# Q.1
drop procedure if exists myProc;
delimiter //
create procedure myProc(in input_code char(3), in input_district char(20))
begin 
	select * from city where countryCode = input_code and district = input_district;
end //
delimiter ;

call myProc('usa', 'california');

# Q.2
drop procedure if exists myProc2;
delimiter //
create procedure myProc2(in input char(20))
begin
	declare existedCountry int;
    declare existedCity int;
    select count(*) into existedCountry from country where name = input;
    select count(*) into existedCity from city where name = input;
    
	if (existedCountry) then
		select format(population, 0) as result from country where name = input;
	elseif (existedCity) then
		select format(population, 0) as result from city where name = input;
	else 
		select '테이블에 없는 국가명(도시명)입니다.' as result;
	end if;
end //
delimiter ;

call myProc2('south korea');
call myProc2('pusan');
call myProc2('pororo');


# Q.3
drop function if exists myFunc;
delimiter //
create function myFunc(data_length int) returns int
begin
    return data_length / 16384;
end //
delimiter ;

select table_name, myFunc(data_length) as 데이터페이지수
from information_schema.tables
where table_schema = 'world';

show table status like 'country';