"use server";

import { z } from "zod";
import {redirect} from "next/navigation";
import {cookies} from "next/headers";
import axiosInterceptorInstance from "@/axiosInterceptorInstance";
import axios from "axios";


type RegisterInfo = {
    messages: string;
}
type SuccessResponse = {
    "token": {
        "refresh": string,
        "access": string
    },
    "message": string,
    "status": {
        "code": number,
        "msg": string
    }
}


// merge SuccessResponse and RegisterInfo
type RegisterResponse = SuccessResponse & RegisterInfo

const login: (email: string, password: string) => Promise<axios.AxiosResponse<SuccessResponse>> = async (email: string, password: string) => {
    try {
        const response: axios.AxiosResponse<SuccessResponse> = await axiosInterceptorInstance.post( '/login', {email,password})
        return response
    } catch (e) {
        if (e.message) {
            return { message: e.message, status: 500 }
        }
        return { status: 500 , message: 'failed to login'}
    }
}

export const  signInApp = async (email: string, password: string) => {
    const parsedCredentials = z
        .object({ email: z.string().email(), password: z.string() })
        .safeParse({ email, password });
    if (!parsedCredentials.success) {
        return { message: 'Invalisd credentials' , token: null};
    }
    const credentials = parsedCredentials.data;
    let response = await login(credentials.email, credentials.password);
    if (response.status === 200) {
        const json = await response.data
        cookies().set('token', json.token.access)
        return { message: 'success', token: json.token.access };
    }
    return { message: 'Invalid credentials', token: null };

}

export async function createUser(
    prevState: {
        message: string;
    },
    formData: FormData,
):  Promise<any> {
    const schema = z.object({
        email: z.string().email({message: "Invalid email"}),
        password: z.string().min(8, {message: "Password must be at least 8 characters long"}),
        confirmPassword: z.string().min(8, {message: "Password must be at least 8 characters long"}),
        name: z.string().min(3, {message: "Name must be at least 3 characters long"})
    });
    const parse = schema.safeParse({
        email: formData.get("email"),
        password: formData.get("password"),
        confirmPassword: formData.get("confirmPassword"),
        name: formData.get("name"),
    });

    if (!parse.success) {
        return {message: parse.error.errors[0].message};
    }

    const data = parse.data;
    try {
        const response: axios.AxiosResponse<RegisterResponse>   = await axiosInterceptorInstance.post('/register', {
            email: data.email,
            username: data.email,
            password: data.password,
            name: data.name
        })
        const messages = response?.data?.messages;
        if (messages) {
            return {message: Object.entries(messages).map(([key, value]) => `${key}: ${value}`).join(', ')}
        }
        //save token from response
        cookies().set('token', response.data.token.access)
    } catch (e) {
        console.log(e)
        return {message: 'failed to create user try later'}
    }
    redirect("/login");
}

export async function authenticate(
    prevState: string | undefined,
    formData: FormData,
) {

    const response = await signInApp(formData.get("email") as string, formData.get("password") as string)
    if (response.token !== null) {
        cookies().set('token', response.token)
        redirect("/home/catalog");
    }
    return { message: response.message };
}
