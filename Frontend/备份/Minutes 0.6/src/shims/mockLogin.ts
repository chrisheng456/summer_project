import Mock from 'mockjs'

// 模拟登录接口
Mock.mock('/api/login', 'post', (options: { body: string }) => {
  const { username, password } = JSON.parse(options.body)

  // 简单模拟：账号 admin 密码 123456 登录成功
  if (username === 'admin' && password === '123456') {
    return {
      code: 200,
      message: 'Login successful',
      token: Mock.Random.guid(),
      userInfo: {
        username: 'admin',
        role: 'admin',
      }
    }
  } else {
    return {
      code: 401,
      message: 'Invalid username or password',
    }
  }
})