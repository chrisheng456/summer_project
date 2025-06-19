// 第一步：创建一个路由器并暴露出去
import {createRouter, createWebHistory, RouterLink} from 'vue-router'//创建路由器，引入createRouter
//导入一个个需要呈现的组件
import logInPage from '@/pages/logInPage.vue'
import translatePage from '@/pages/translatePage.vue'
import meetingSummary from '@/pages/meetingSummary.vue'
//  第二步：创建路由器
const router =createRouter({
    history:createWebHistory(),//设定路由器的工作模式，这里使用的是history模式，//后端需要配合配置路径
    routes:[//编写一个个路由规则
    {
        path:'/logInPage',
        component:logInPage
    },
    {
        path:'/translatePage',
        component:translatePage
    },
    {
        path:'/meetingSummary',
        component:meetingSummary
    }

    ]
})
export default router
    

