
import axios from 'axios';
import {cookies} from "next/headers";

const axiosInterceptorInstance = axios.create({
    baseURL: 'http://34.227.92.112:8000/api',

});

axiosInterceptorInstance.interceptors.request.use(
    (config) => {
        // Handle request headers here
        const token = cookies().get('token').value;
        console.table(
            config.data
        )
        if (token && config.url !== '/login' && config.url !== '/register') {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        // Handle request errors here
        return Promise.reject(error);
    }
);

axiosInterceptorInstance.interceptors.response.use(
    (response) => {
        /*console.table({
            data: response.data,
            url: response.config.url,
            status: response.status,
        });*/
        return response;
    },
    (error) => {
/*        console.table(
            {
                url: error.config.url,
                status: error.response.status,
                data: error.response.data,
            }
        )*/
        return Promise.reject(error);
    }
);


export default axiosInterceptorInstance;
