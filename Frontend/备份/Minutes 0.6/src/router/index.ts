// 第一步：创建一个路由器并暴露出去
import {createRouter, createWebHistory, RouterLink} from 'vue-router'//创建路由器，引入createRouter
//导入一个个需要呈现的组件
import loginPage from '@/pages/loginPage.vue'
import UploadHistory from '@/pages/UploadHistory.vue'
import MeetingNotes from '@/pages/MeetingNotes.vue'
import Register from '@/pages/Register.vue'
import ForgotPassword from "@/pages/ForgotPassword.vue";


//  第二步：创建路由器
const routes =
    [//编写一个个路由规则
    {
    path:'/',
    redirect:'/loginPage'
    },

    {
    path: '/loginPage',  
    name: 'LoginPage',
    component: loginPage
    },


    {
    name:"MeetingNotes",
    path:'/MeetingNotes',
    component:MeetingNotes
    },
    {
    name:"Register",
    path:'/Register',
    component:Register
    },

    {
    path: "/ForgotPassword",
    name: "ForgotPassword",
    component: ForgotPassword
    },

    {   
    name:"UploadHistory",
    path:'/UploadHistory',
    component:UploadHistory
    }

    ]


const router = createRouter({
  history: createWebHistory(),
  routes
})


export default router
    

