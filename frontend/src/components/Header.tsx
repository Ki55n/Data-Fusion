"use client";
import React, { useEffect, useState } from "react";
import Image from 'next/image';
import { UserAuth } from "../app/context/AuthContext";
import { FaUserCircle } from 'react-icons/fa';
import Spinner from "./Spinner";
import { useRouter } from "next/navigation";


export default function Header({ users }) {
    const { user, googleSignIn, logOut } = UserAuth();
  const [loading, setLoading] = useState(true);
const router = useRouter();
 

  const handleSignOut = async () => {
    try {
      await logOut();
    } catch (error) {
      console.log(error);
    }
  };

  useEffect(() => {
    const checkAuthentication = async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
      setLoading(false);
    };
    checkAuthentication();
  }, [user]);
    return (
        <nav className='w-full pl-5 pr-9   flex justify-between items-center bg-black shadow-md sticky top-0 z-50'>
            <Image src='/logo.png' width={105} height={60} alt="logo" />
            <div className='flex items-center'>
                {loading ? <Spinner /> : user ? (
                    <div className='flex items-center'>
                        <FaUserCircle className='text-white text-3xl mr-2' />
                        <p className='text-white font-bold'>{user.displayName}</p>
                        <button className='text-white font-bold ml-2' onClick={handleSignOut}>Logout</button>
                    </div>
                ) : (
                    router.push('/login')
                )}
            </div>
        </nav>
    );
}
