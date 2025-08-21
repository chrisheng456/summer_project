
// Import createApp to create the application
import { createApp } from 'vue'
// Import the root component App
import App from './App.vue'
// Import the router
import router from './router'
// If you are using Element Plus:
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
// Import Pinia
import { createPinia } from 'pinia'
// Load mock files only in development environment


// Create an app instance
const app = createApp(App)
const pinia = createPinia()

app.use(router) // Install plugins: router and Element Plus (or other UI libraries)
app.use(ElementPlus)
app.use(pinia) // Pass the result of createPinia() directly
app.mount('#app') // Mount the app to the element with id="app"
