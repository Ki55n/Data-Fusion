import type { Metadata } from 'next'
import Aside from "@/components/Aside";
import React from "react";
import Header from "@/components/Header";
import Providers from "@/app/providers";
import {Button} from "@nextui-org/button";
import { FaMagic } from "react-icons/fa";
import Chatbot from "@/components/chatbot";
export const metadata: Metadata = {
    title: 'Data fusion',
    description: 'Data fusion',
}
export default function RootLayout({
    children
}: {
    children: React.ReactNode
}) {
    const user = {
        name: 'John Doe',
        email: 'john.doe@example.com',
        avatar: '/path/to/avatar.jpg'
    };
    return (
        <html >
            <body className="dark text-foreground bg-background" >
                <Providers>
                    <main className="dark text-foreground bg-background">
                    <Header user={user} />
                    <div className="flex h-screen">
                        <Aside />
                        {children}
                        <div className='fixed bottom-10 left-1/2 transform -translate-x-1/2 bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-700'><Chatbot /></div>
                    </div>
                    </main>
                </Providers>
            </body>
        </html>

    )
}



