# 기본 DB

DROP TABLE IF EXISTS followers;
DROP TABLE IF EXISTS friendlist;
DROP TABLE IF EXISTS friendships;
DROP TABLE IF EXISTS mainhome;
DROP TABLE IF EXISTS miniroom;
DROP TABLE IF EXISTS photo_comments;
DROP TABLE IF EXISTS post_comments;
DROP TABLE IF EXISTS report;
DROP TABLE IF EXISTS visitorlog;
DROP TABLE IF EXISTS guestbook_comments;
DROP TABLE IF EXISTS guestbook;
DROP TABLE IF EXISTS photo;
DROP TABLE IF EXISTS post;
DROP TABLE IF EXISTS member;

CREATE TABLE `member` (
  `mem_num` int NOT NULL AUTO_INCREMENT,
  `mem_id` varchar(50) NOT NULL,
  `mem_pw` varchar(50) NOT NULL,
  `nickname` varchar(50) DEFAULT NULL,
  `is_admin` int DEFAULT 0,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`mem_id`),
  UNIQUE KEY `mem_id` (`mem_id`),
  UNIQUE KEY `mem_num` (`mem_num`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `followers` (
  `auto_num` int NOT NULL AUTO_INCREMENT,
  `follower_id` varchar(50) NOT NULL,
  `following_id` varchar(50) NOT NULL,
  `create_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`auto_num`),
  KEY `follower_id` (`follower_id`),
  KEY `following_id` (`following_id`),
  CONSTRAINT `followers_ibfk_1` FOREIGN KEY (`follower_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `followers_ibfk_2` FOREIGN KEY (`following_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `friendlist` (
  `auto_num` int NOT NULL AUTO_INCREMENT,
  `mem_id` varchar(50) NOT NULL,
  `friend_id` varchar(50) NOT NULL,
  `create_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`auto_num`),
  KEY `mem_id` (`mem_id`),
  KEY `friend_id` (`friend_id`),
  CONSTRAINT `friendlist_ibfk_1` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `friendlist_ibfk_2` FOREIGN KEY (`friend_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `friendships` (
  `auto_num` int NOT NULL AUTO_INCREMENT,
  `mem_id` varchar(50) NOT NULL,
  `friend_id` varchar(50) NOT NULL,
  `status` enum('pending','accepted') DEFAULT 'pending',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`auto_num`),
  KEY `mem_id` (`mem_id`),
  KEY `friend_id` (`friend_id`),
  CONSTRAINT `friendships_ibfk_1` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `friendships_ibfk_2` FOREIGN KEY (`friend_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `guestbook` (
  `guestbook_id` int NOT NULL AUTO_INCREMENT,
  `mem_id` varchar(50) NOT NULL,
  `visitor_id` varchar(50) NOT NULL,
  `content` text,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`guestbook_id`),
  KEY `mem_id` (`mem_id`),
  KEY `visitor_id` (`visitor_id`),
  CONSTRAINT `owner_id` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `visitor_id` FOREIGN KEY (`visitor_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `guestbook_comments` (
  `comment_id` int NOT NULL AUTO_INCREMENT,
  `guestbook_id` int NOT NULL,
  `mem_id` varchar(50) NOT NULL,
  `comment_text` text,
  `parent_comment_id` int DEFAULT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`comment_id`),
  KEY `guestbook_id` (`guestbook_id`),
  KEY `mem_id` (`mem_id`),
  KEY `parent_comment_id` (`parent_comment_id`),
  CONSTRAINT `guestbook_comments_ibfk_1` FOREIGN KEY (`guestbook_id`) REFERENCES `guestbook` (`guestbook_id`) ON DELETE CASCADE,
  CONSTRAINT `guestbook_comments_ibfk_2` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `guestbook_comments_ibfk_3` FOREIGN KEY (`parent_comment_id`) REFERENCES `guestbook_comments` (`comment_id`) ON DELETE CASCADE
);

CREATE TABLE `mainhome` (
  `mem_id` varchar(50) NOT NULL,
  `profile_picture` varchar(255) DEFAULT NULL,
  `profile_bio` text,
  `title` varchar(100) DEFAULT NULL,
  `count_today` int DEFAULT '0',
  `count_total` int DEFAULT '0',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT `mainhome_ibfk_1` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `miniroom` (
  `auto_num` int NOT NULL AUTO_INCREMENT,
  `mem_id` varchar(50) NOT NULL,
  `photo_url` varchar(255) DEFAULT NULL,
  `content` text,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`auto_num`),
  KEY `mem_id` (`mem_id`),
  CONSTRAINT `miniroom_ibfk_1` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `photo` (
  `photo_id` int NOT NULL AUTO_INCREMENT,
  `mem_id` varchar(50) NOT NULL,
  `title` varchar(100) NOT NULL,
  `content` text,
  `photo_url` varchar(1000) NOT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`photo_id`),
  KEY `mem_id` (`mem_id`),
  CONSTRAINT `photo_ibfk_1` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `photo_comments` (
  `comment_id` int NOT NULL AUTO_INCREMENT,
  `photo_id` int NOT NULL,
  `mem_id` varchar(50) NOT NULL,
  `comment_text` text,
  `parent_comment_id` int DEFAULT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`comment_id`),
  KEY `photo_id` (`photo_id`),
  KEY `mem_id` (`mem_id`),
  KEY `parent_comment_id` (`parent_comment_id`),
  CONSTRAINT `photo_comments_ibfk_1` FOREIGN KEY (`photo_id`) REFERENCES `photo` (`photo_id`) ON DELETE CASCADE,
  CONSTRAINT `photo_comments_ibfk_2` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `photo_comments_ibfk_3` FOREIGN KEY (`parent_comment_id`) REFERENCES `photo_comments` (`comment_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `post` (
  `post_id` int NOT NULL AUTO_INCREMENT,
  `mem_id` varchar(50) NOT NULL,
  `title` varchar(100) NOT NULL,
  `content` text,
  `post_date` datetime,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`post_id`),
  KEY `mem_id` (`mem_id`),
  CONSTRAINT `post_ibfk_1` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `post_comments` (
  `post_id` int NOT NULL,
  `comment_id` varchar(50) NOT NULL,
  `comment_text` text,
  `parent_comment_id` varchar(50) DEFAULT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`comment_id`),
  KEY `post_id` (`post_id`),
  KEY `comment_id` (`comment_id`),
  KEY `parent_comment_id` (`parent_comment_id`),
  CONSTRAINT `post_comments_ibfk_1` FOREIGN KEY (`post_id`) REFERENCES `post` (`post_id`) ON DELETE CASCADE,
  CONSTRAINT `post_comments_ibfk_2` FOREIGN KEY (`parent_comment_id`) REFERENCES `post` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `post_comments_ibfk_3` FOREIGN KEY (`parent_comment_id`) REFERENCES `post_comments` (`comment_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `report` (
  `report_id` int NOT NULL AUTO_INCREMENT,
  `reporter_id` varchar(50) NOT NULL,
  `reported_id` varchar(50) NOT NULL,
  `reason` text NOT NULL,
  `report_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`report_id`),
  KEY `reporter_id` (`reporter_id`),
  KEY `reported_id` (`reported_id`),
  CONSTRAINT `report_ibfk_1` FOREIGN KEY (`reporter_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `report_ibfk_2` FOREIGN KEY (`reported_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `visitorlog` (
  `log_id` int NOT NULL AUTO_INCREMENT,
  `mem_id` varchar(50) NOT NULL,
  `visitor_id` varchar(50) NOT NULL,
  `visited_at` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`log_id`),
  KEY `mem_id` (`mem_id`),
  KEY `visitor_id` (`visitor_id`),
  CONSTRAINT `visitorlog_ibfk_1` FOREIGN KEY (`mem_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE,
  CONSTRAINT `visitorlog_ibfk_2` FOREIGN KEY (`visitor_id`) REFERENCES `member` (`mem_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


# ====================================
# 1.
DROP PROCEDURE IF EXISTS GetUpdatedPostPhoto;
DELIMITER //
CREATE PROCEDURE GetUpdatedPostPhoto(IN input_mem_id varchar(50))
BEGIN
    SELECT '다이어리' AS table_name, title, created_at 
    FROM post
    WHERE mem_id = input_mem_id
    UNION ALL
    SELECT '사진첩' AS table_name, title, created_at 
    FROM photo
    WHERE mem_id = input_mem_id
    ORDER BY created_at DESC
    LIMIT 4;
END //
DELIMITER ;

CALL GetUpdatedPostPhoto(1);


# 2.
DROP PROCEDURE IF EXISTS GetCount;
DELIMITER //
CREATE PROCEDURE GetCount(IN input_mem_id varchar(50))
BEGIN
    SELECT
        (SELECT COUNT(*) FROM post WHERE mem_id = input_mem_id) AS post_count,
        (SELECT COUNT(*) FROM photo WHERE mem_id = input_mem_id) AS photo_count,
        (SELECT COUNT(*) FROM guestbook WHERE mem_id = input_mem_id) AS guestbook_count;
END //
DELIMITER ;

CALL GetCount(1);

# 3.
DROP PROCEDURE IF EXISTS GetFollow;
DELIMITER //
CREATE PROCEDURE GetFollow(IN input_mem_id varchar(50))
BEGIN
    SELECT 
		(SELECT count(*) FROM friendships WHERE mem_id = input_mem_id) AS follower,
		(SELECT count(*) FROM friendships WHERE friend_id = input_mem_id) AS following;
END //
DELIMITER ;

CALL GetFollow(1);

# 4.
DROP PROCEDURE IF EXISTS GetAllUser;
DELIMITER //
CREATE PROCEDURE GetAllUser(IN userId VARCHAR(50))
BEGIN
    SELECT 
        m.mem_id, 
        mh.profile_picture,
        CASE 
            WHEN f.friend_id IS NOT NULL THEN 1
            ELSE 0
        END AS follow
    FROM 
        member m
    JOIN 
        mainhome mh ON m.mem_id = mh.mem_id
    LEFT JOIN 
        friendships f ON m.mem_id = f.friend_id AND f.mem_id = userId
    WHERE 
        m.mem_id <> userId OR userId IS NULL; -- userId와 mem_id가 다른 경우만 출력
END //
DELIMITER ;


# 5.
DROP PROCEDURE IF EXISTS GetUser;
DELIMITER //
CREATE PROCEDURE GetUser(IN userId VARCHAR(50), IN id VARCHAR(50))
BEGIN
    SELECT 
        m.mem_id, 
        mh.profile_picture,
        CASE 
            WHEN f.friend_id IS NOT NULL THEN 1
            ELSE 0
        END AS follow
    FROM 
        member m
    JOIN 
        mainhome mh ON m.mem_id = mh.mem_id
    LEFT JOIN 
        friendships f ON m.mem_id = f.friend_id AND f.mem_id = userId
    WHERE 
        m.mem_id LIKE CONCAT('%', id, '%') AND m.mem_id <> userId;
END //
DELIMITER ;

# 6.
DROP PROCEDURE IF EXISTS GetAllFriend;
delimiter //
CREATE PROCEDURE GetAllFriend(IN userId VARCHAR(50), IN id VARCHAR(50))
BEGIN
	SELECT 
		f.friend_id,
		mh.profile_picture,
		CASE 
			WHEN EXISTS (
				SELECT 1
				FROM friendships
				WHERE mem_id = userId AND friend_id = f.friend_id
			) THEN 1
			ELSE 0
		END AS follow
	FROM 
		friendships f
	JOIN 
		mainhome mh ON f.friend_id = mh.mem_id
	WHERE 
		f.mem_id = id;
END //
delimiter ;

# 7.
DROP PROCEDURE IF EXISTS GetRequestedFriend;
DELIMITER //
CREATE PROCEDURE GetRequestedFriend(IN userId VARCHAR(50), IN id VARCHAR(50))
BEGIN
    SELECT 
        f.mem_id AS friend_id,
        mh.profile_picture,
        CASE 
            WHEN EXISTS (
                SELECT 1
                FROM friendships
                WHERE mem_id = userId AND friend_id = f.friend_id  AND status = 'pending'
            ) THEN 1
            ELSE 0
        END AS follow
    FROM 
        friendships f
    JOIN 
        mainhome mh ON f.mem_id = mh.mem_id
    WHERE 
        f.friend_id = userId;
END //
DELIMITER ;


# today 리셋하는 event 생성
DROP EVENT IF EXISTS reset_count_today;
DELIMITER //
CREATE EVENT reset_count_today
ON SCHEDULE EVERY 1 DAY
STARTS TIMESTAMP(CURRENT_DATE + INTERVAL 1 DAY)
DO
BEGIN
    UPDATE mainhome SET count_today = 0;
END //
DELIMITER ;

show events;

