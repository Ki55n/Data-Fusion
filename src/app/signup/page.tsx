"use client";
import styles from "./signup.module.css";
import { useFormState } from "react-dom";
import { createUser } from "@/actions/account";
import { useState } from "react";
import { useCreateUserWithEmailAndPassword } from "react-firebase-hooks/auth";
import { auth } from "@/app/firebase/config";
import { useRouter } from "next/navigation";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [createUserWithEmailAndPassword] =
    useCreateUserWithEmailAndPassword(auth);
  const router = useRouter();
  const handleSignUp = async () => {
    try {
      const res = await createUserWithEmailAndPassword(email, password);
      console.log({ res });
      sessionStorage.setItem("user", String(true));
      setEmail("");
      setPassword("");
      router.push("/home/catalog");
    } catch (e) {
      console.error(e);
    }
  };
  const initialState = {
    message: "",
  };
  const [state, formAction] = useFormState(createUser, initialState);

  return (
    <div className={styles.container}>
      <div className={styles.logoWrapper}>
        <img src="/logo.png" alt="Logo" className={styles.logo} />
      </div>
      <div className={styles.formWrapper}>
        <h1 className={styles.header}>Sign Up</h1>
        <label htmlFor="name" className={styles.inputLabel}>
          Name:
        </label>
        <input
          type="text"
          placeholder="Name"
          id="name"
          className="w-full p-3 mb-4 bg-gray-700 rounded outline-none text-white placeholder-gray-500"
          name={"name"}
          required
        />
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
          onClick={handleSignUp}
          className="w-full p-3 bg-indigo-600 rounded text-white hover:bg-indigo-500"
        >
          Sign Up
        </button>
        <div className="flex justify-between items-center mt-4">
          <div className="text-sm text-500">
            Already have an account?
          </div>
          <a href="/login" className="text-sm text-blue-500">
            Log In
          </a>
          </div>
      </div>
    </div>
  );
}
