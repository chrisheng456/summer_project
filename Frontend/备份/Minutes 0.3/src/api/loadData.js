import axios from 'axios';

/**
 * 异步获取远程数据的通用函数，使用 Axios 发送 GET 请求。
 *
 * @async
 * @function fetchData
 * @param {string} url - 请求的 URL。
 * @param {object} [config={}] - Axios 请求配置对象（可选）。
 * @returns {Promise<any|null>} 请求成功返回响应数据，失败返回 null。
 *
 * @example
 * const data = await fetchData('https://api.example.com/data');
 * 使用默认导出，在其他文件导入直接使用 import
 */
export default async function fetchData(url, config = {}) {
    try {
        const response = await axios.get(url, config);
        return response.data;
    } catch (error) {
        console.error('Request error:', error);
        return null;
    }
}
