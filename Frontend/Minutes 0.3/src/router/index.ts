// 第一步：创建一个路由器并暴露出去
import {createRouter, createWebHistory, RouterLink} from 'vue-router'//创建路由器，引入createRouter
//导入一个个需要呈现的组件
import LoginPage from '@/pages/LoginPage.vue'
import UploadHistory from '@/pages/UploadHistory.vue'
import MeetingNotes from '@/pages/MeetingNotes.vue'

//  第二步：创建路由器
const router =createRouter({
    history:createWebHistory(),//设定路由器的工作模式，这里使用的是history模式，//后端需要配合配置路径
    routes:[//编写一个个路由规则
    {
        name:"LoginPage",
        path:'/LoginPage',
        component:LoginPage
    },

   {
        name:"MeetingNotes",
        path:'/MeetingNotes',
        component:MeetingNotes
    },
 

    {   name:"UploadHistory",
        path:'/UploadHistory',
        component:UploadHistory
    }

    ]
})
export default router
    

