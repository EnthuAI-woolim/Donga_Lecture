delete from member;
ALTER TABLE member AUTO_INCREMENT = 1;
delete from mainhome;
delete from miniroom;
ALTER TABLE miniroom AUTO_INCREMENT = 1;
delete from post;
ALTER TABLE post AUTO_INCREMENT = 1;
delete from photo;
ALTER TABLE photo AUTO_INCREMENT = 1;
delete from guestbook;
ALTER TABLE guestbook AUTO_INCREMENT = 1;


select * from member;
select * from mainhome;
select * from miniroom;
select * from post;
select * from photo;
select * from guestbook;

desc friendships;

call GetCount(1);
call GetUpdatedPostPhoto(1);

UPDATE mainhome
SET profile_picture = 'https://cdn.pixabay.com/photo/2020/09/04/20/09/cartoon-5544856_1280.jpg'
WHERE auto_mem_id = 1;


SELECT * FROM member WHERE mem_id = '1';
desc friendlist;


INSERT INTO `cyworld`.`friendlist`(
`user_id`,
`friend_id`)
VALUES
(
'1',
'2');

select friend_id from friendlist where user_id = '1';

select * from friendlist;
delete from friendships where user_id = '1';


select * from member;
select auto_mem_id from member where mem_id = ?;

SELECT m.mem_id, mh.profile_picture
FROM member m
JOIN mainhome mh ON m.auto_mem_id = mh.auto_mem_id;

select * from friendlist;
desc friendlist;	
desc mainhome;
desc friendships;

update mainhome
set profile_picture = 'https://cdn.pixabay.com/photo/2020/09/04/20/09/cartoon-5544856_1280.jpg';

SELECT f.friend_id, mh.profile_picture 
FROM friendlist f
JOIN mainhome mh 
ON m.auto_mem_id = mh.auto_mem_id 
WHERE f.user_id = ?;

select friend_id from friendlist where user_id = '1';

SELECT m.mem_id, mh.profile_picture FROM member m JOIN mainhome mh ON m.auto_mem_id = mh.auto_mem_id WHERE m.mem_id = '1';

select * from mainhome;
select * from post;
select * from report;

select title, content, post_date from post where mem_id = '1';
INSERT INTO post (mem_id, title, content, post_date) VALUES (?, ?, ?, ?);

select mem_id, profile_picture from mainhome where mem_id = (select friend_id from friendlist where mem_id = '1');

select friend_id from friendlist where mem_id = '1';

SELECT mem_id, profile_picture FROM mainhome WHERE mem_id = (SELECT friend_id FROM friendships WHERE mem_id = '1');
select * from miniroom;

desc friendships;

SELECT title, content, photo_url FROM photo WHERE mem_id = ?;
# vid, con, crea
select visitor_id, content, created_at from guestbook where mem_id = ?;

UPDATE friendships SET status = 'accepted' WHERE mem_id = '1' and friend_id = '2';

select * from friendships;
INSERT INTO friendships (mem_id, friend_id) values('2', '1');

select count(*) as follower from friendships where mem_id = '1';
select count(*) as following from friendships where friend_id = '1';

select * from mainhome;

select * from member;
select * from friendlist;
insert into friendlist (mem_id, friend_id) values ('9', '8');

SELECT mem_id, profile_picture FROM mainhome WHERE mem_id = (SELECT friend_id FROM friendlist WHERE mem_id = '1');


select * from friendlist;

drop procedure if exists GetAllFriend;
DELIMITER $$

CREATE PROCEDURE `GetAllFriend`(IN userId VARCHAR(50))
BEGIN
    SELECT 
        DISTINCT CASE
            WHEN f.mem_id = userId THEN f.friend_id
            WHEN f.friend_id = userId THEN f.mem_id
        END AS friend_id,
        mh.profile_picture,
        1 AS follow
    FROM 
        friendlist f
    JOIN 
        member m ON (m.mem_id = f.friend_id OR m.mem_id = f.mem_id)
    JOIN 
        mainhome mh ON m.mem_id = mh.mem_id
    WHERE 
        f.mem_id = userId OR f.friend_id = userId;
END$$

DELIMITER ;



call GetAllFriend('8');

select * from friendlist;

