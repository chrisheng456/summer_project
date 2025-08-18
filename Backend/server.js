// server.js
const express = require('express');
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const app = express();
const db = new sqlite3.Database('./database.db');
const JWT_SECRET = 'your-very-secure-secret'; // 生产环境请用更复杂的密钥

// 中间件
app.use(cors());
app.use(express.json());

// 用户注册
app.post('/api/register', async (req, res) => {
  const { username, email, password } = req.body;

  try {
    // 检查用户名是否已存在
    db.get('SELECT id FROM users WHERE username = ?', [username], async (err, row) => {
      if (err) throw err;
      if (row) return res.status(400).json({ error: '用户名已存在' });

      // 加密密码
      const hashedPassword = await bcrypt.hash(password, 10);
      
      // 插入新用户
      db.run(
        'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
        [username, email, hashedPassword],
        function(err) {
          if (err) throw err;
          res.json({ 
            id: this.lastID,
            username,
            email
          });
        }
      );
    });
  } catch (error) {
    res.status(500).json({ error: '服务器错误' });
  }
});

// 用户登录
app.post('/api/login', (req, res) => {
  const { username, password } = req.body;

  db.get('SELECT * FROM users WHERE username = ?', [username], async (err, user) => {
    if (err) return res.status(500).json({ error: '数据库错误' });
    if (!user) return res.status(401).json({ error: '用户不存在' });

    // 验证密码
    const isValid = await bcrypt.compare(password, user.password_hash);
    if (!isValid) return res.status(401).json({ error: '密码错误' });

    // 生成JWT令牌
    const token = jwt.sign(
      { id: user.id, username: user.username },
      JWT_SECRET,
      { expiresIn: '1h' }
    );

    res.json({ 
      token,
      user: {
        id: user.id,
        username: user.username,
        email: user.email
      }
    });
  });
});

// 启动服务器
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
});