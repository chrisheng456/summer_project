-- =====================================================
-- meeting_schema_updated.sql
-- 完整的数据库结构定义脚本（含 action、decision、conflict）
-- =====================================================

-- 1. 如果表已存在，先删了重建（可选）
DROP TABLE IF EXISTS speech_segment;
DROP TABLE IF EXISTS agenda_line;
DROP TABLE IF EXISTS agenda_item;
DROP TABLE IF EXISTS attendee;
DROP TABLE IF EXISTS event;

-- =====================================================
-- 2. 创建 event 表
-- =====================================================
CREATE TABLE event (
  id INT PRIMARY KEY,
  name VARCHAR(200),
  date DATETIME,
  start_time DATETIME,
  location VARCHAR(200)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- =====================================================
-- 3. 创建 attendee 表
-- =====================================================
CREATE TABLE attendee (
  id INT PRIMARY KEY,
  event_id INT NOT NULL,
  name VARCHAR(100),
  attending BOOLEAN,
  userCanEdit BOOLEAN,
  FOREIGN KEY (event_id) REFERENCES event(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- =====================================================
-- 4. 创建 agenda_item 表（含分类与摘要字段）
-- =====================================================
CREATE TABLE agenda_item (
  id INT PRIMARY KEY,
  event_id INT NOT NULL,
  number VARCHAR(10),
  title VARCHAR(200),
  owner VARCHAR(100),
  label VARCHAR(50),
  label_score FLOAT,
  summary TEXT,
  explanation TEXT,
  FOREIGN KEY (event_id) REFERENCES event(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- =====================================================
-- 5. 创建 agenda_line 表（原始逐行发言）
-- =====================================================
CREATE TABLE agenda_line (
  id INT AUTO_INCREMENT PRIMARY KEY,
  agenda_item_id INT NOT NULL,
  start_time FLOAT,
  end_time FLOAT,
  speaker VARCHAR(100),
  `text` TEXT,
  FOREIGN KEY (agenda_item_id) REFERENCES agenda_item(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- =====================================================
-- 6. 创建 speech_segment 表（每段发言片段）
--    —— 包含 summary、action、decision、conflict
-- =====================================================
CREATE TABLE speech_segment (
  id INT AUTO_INCREMENT PRIMARY KEY,
  agenda_item_id INT NOT NULL,
  speaker VARCHAR(100),
  start_time FLOAT,
  end_time FLOAT,
  `text` TEXT,
  summary TEXT,
  action TEXT,
  decision TEXT,
  conflict BOOLEAN DEFAULT FALSE,
  FOREIGN KEY (agenda_item_id) REFERENCES agenda_item(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
