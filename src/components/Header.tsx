import Image from 'next/image';
import React from 'react';
import { FaUserCircle } from 'react-icons/fa';

export default function Header({ user }) {
    return (
        <nav className='w-full pl-5 pr-9   flex justify-between items-center bg-black shadow-md sticky top-0 z-50'>
            <Image src='/logo.png' width={105} height={60} alt="logo" />
            <div className='flex border-spacing-2  rounded items-center space-x-4'>
                <div className='text-sm'>
                    <p className='font-semibold text-center  text-blue'>{user.name}</p>
                    <p className='text-gray-500'>{user.email}</p>
                </div>
                <FaUserCircle className='text-4xl  text-blue-700' />
            </div>
        </nav>
    );
}
