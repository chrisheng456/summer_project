import { http } from '../http'
import type {LoginReq, LoginResp} from '@/types';

export const authApi = {
  login: async (payload: LoginReq): Promise<LoginResp> => {
    const { username, password } = payload;
    const resp = await http.post<LoginResp>(
      "/auth/login",
      { username, password },              // body
    );
    return resp.data;
  },
};