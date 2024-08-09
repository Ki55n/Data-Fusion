"use client";
import styles from "./login.module.css";
import { useState } from "react";
import { useFormState } from "react-dom";
import { authenticate } from "@/actions/account";
import { useSignInWithEmailAndPassword } from "react-firebase-hooks/auth";
import { auth } from "@/app/firebase/config";
import { useRouter } from "next/navigation";
import { GoogleAuthProvider } from "firebase/auth";
import { signInWithPopup } from "firebase/auth";
export default function Login() {
  const initialState = {
    message: "",
  };
  //  const [state, formAction] = useFormState(authenticate, initialState);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [signInWithEmailAndPassword] = useSignInWithEmailAndPassword(auth);
  const router = useRouter();
 

  const handleSignIn = async () => {
    try {
      const res = await signInWithEmailAndPassword(email, password);
      console.log({ res });
      sessionStorage.setItem("user", String(true));
      setEmail("");
      setPassword("");
      router.push("/home/catalog");
    } catch (e) {
      console.error(e);
    }
  };
  const handlegoogle= async(e)=>{
    const provider = await new GoogleAuthProvider();
   
   return signInWithPopup(auth, provider)
    

  }

  return (
    <div className={styles.container}>
      <div className={styles.logoWrapper}>
        <img src="/logo.png" alt="Logo" className={styles.logo} />
      </div>
      <div className={styles.formWrapper}>
        <h1 className={styles.header}>Login</h1>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full p-3 mb-4 bg-gray-700 rounded outline-none text-white placeholder-gray-500"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full p-3 mb-4 bg-gray-700 rounded outline-none text-white placeholder-gray-500"
        />
        <button
          onClick={handleSignIn}
          className="w-full p-3 bg-indigo-600 rounded text-white hover:bg-indigo-500"
        >
          Sign In
        </button>

        {/* Google Login Button */}
        <div class="text-center">or</div>
        <div className="mt-4">
          <button onclick={handlegoogle} className="flex items-center justify-center w-full p-2 text-white bg-red-600 rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2">
            <svg
              className="w-5 h-5 mr-2"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M22.674 11.24c0-.748-.068-1.464-.187-2.154H12v4.07h5.961c-.257 1.29-.998 2.381-2.112 3.109v2.566h3.418c2.006-1.85 3.156-4.569 3.156-7.591z"
                fill="#4285F4"
              />
              <path
                d="M12 23c3.106 0 5.705-1.034 7.606-2.806l-3.418-2.566c-.948.635-2.145 1.015-3.569 1.015-2.749 0-5.073-1.856-5.906-4.348H4.2v2.735C6.089 20.975 8.848 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M6.094 14.295a6.95 6.95 0 01-.365-2.295c0-.796.133-1.565.365-2.295V6.97H4.2a10.972 10.972 0 000 9.86l1.894-2.535z"
                fill="#FBBC05"
              />
              <path
                d="M12 4.856c1.512 0 2.868.52 3.937 1.542l2.936-2.936C16.59 1.977 14.284 1 12 1 8.848 1 6.089 3.025 4.2 6.971l1.894 2.735c.834-2.492 3.158-4.348 5.906-4.348z"
                fill="#EA4335"
              />
            </svg>
            Sign in with Google
          </button>
        </div>

        <div className="flex justify-between items-center mt-4">
          <a href="#" className="text-sm text-blue-500">
            Forgot password?
          </a>
          <a href="/signup" className="text-sm text-blue-500">
            Sign Up
          </a>
        </div>
      </div>
    </div>
  );
}
