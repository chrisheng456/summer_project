
import {createRouter, createWebHistory, RouterLink} from 'vue-router'
import LoginPage from '@/pages/logInPage.vue'
import UploadHistory from '@/pages/UploadHistory.vue'
import MeetingNotes from '@/pages/MeetingNotes.vue'
import Register from '@/pages/Register.vue'
import ForgotPassword from "@/pages/ForgotPassword.vue";



const routes =
    [
    {
    path:'/',
    redirect:'/logInPage'
    },
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
    },
    {
      name: 'TransPage',
      path: '/trans',
      component: () => import('@/components/TransPage.vue') // 或 '@/pages/TransPage.vu
    }
    ]


const router = createRouter({
  history: createWebHistory(),
  routes
})


export default router
    

