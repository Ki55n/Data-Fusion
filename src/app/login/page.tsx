"use client";
import styles from "./login.module.css";
import { useState } from "react";
import { useFormState } from "react-dom";
import { authenticate } from "@/actions/account";
import { useSignInWithEmailAndPassword } from "react-firebase-hooks/auth";
import { auth } from "@/app/firebase/config";
import { useRouter } from "next/navigation";

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
