-- 1. 会议信息表
CREATE TABLE `event` (
  `event_id`   INT          NOT NULL PRIMARY KEY,
  `name`       VARCHAR(255) NOT NULL,
  `date`       DATETIME     NULL,
  `start_time` DATETIME     NULL,
  `location`   VARCHAR(255) NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. 与会者表
CREATE TABLE `attendee` (
  `attendee_id`   INT          NOT NULL PRIMARY KEY,
  `event_id`      INT          NOT NULL,
  `name`          VARCHAR(255) NOT NULL,
  `attending`     TINYINT(1)   NOT NULL DEFAULT 1,
  `user_can_edit` TINYINT(1)   NOT NULL DEFAULT 0,
  FOREIGN KEY (`event_id`) REFERENCES `event` (`event_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. 议程项表
CREATE TABLE `agenda_item` (
  `agenda_id`         INT          NOT NULL PRIMARY KEY,
  `event_id`          INT          NOT NULL,
  `number`            VARCHAR(20)  NULL,
  `title`             VARCHAR(255) NULL,
  `indent`            INT          NULL,
  `calculated_start`  DATETIME     NULL,
  `length_minutes`    INT          NULL,
  `owner`             VARCHAR(255) NULL,
  `label`             VARCHAR(50)  NULL,
  `label_score`       FLOAT        NULL,
  `summary`           TEXT         NULL,
  `explanation`       TEXT         NULL,
  FOREIGN KEY (`event_id`) REFERENCES `event` (`event_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. 发言片段表
CREATE TABLE `speech_segment` (
  `segment_id`   INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `agenda_id`    INT          NOT NULL,
  `speaker`      VARCHAR(128) NOT NULL,
  `start_time`   FLOAT        NOT NULL,
  `end_time`     FLOAT        NOT NULL,
  `text`         TEXT         NOT NULL,
  FOREIGN KEY (`agenda_id`) REFERENCES `agenda_item` (`agenda_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;