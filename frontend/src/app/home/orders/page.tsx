'use client'
import styles from './orderList.module.css';
import { useState, useEffect } from 'react';

const orders =  [
    { name: 'MacBook Pro', date: '2024-01-15', amount: '$2399' },
    { name: 'iPhone 13', date: '2024-02-10', amount: '$799' },
    { name: 'AirPods Pro', date: '2024-03-05', amount: '$249' },
    { name: 'Apple Watch Series 7', date: '2024-04-22', amount: '$399' },
    { name: 'iPad Pro', date: '2024-05-30', amount: '$1099' },
    { name: 'HomePod', date: '2024-06-18', amount: '$299' },
];

export default function OrderList() {
    const [theme, setTheme] = useState('dark');

    // useEffect(() => {
    //     // Check for system theme preference
    //     const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    //     setTheme(prefersDark ? 'dark' : 'light');
    // }, []);

    // const toggleTheme = () => {
    //     setTheme(theme === 'light' ? 'dark' : 'light');
    // };

    return (
        <div className={`${styles.container} ${styles[theme]}`}>
            {/* <button onClick={toggleTheme} className={styles.themeToggle}>
                Toggle Theme
            </button> */}
                <br />
                <br />
            <h1 className="text-4xl font-bold text-center mb-8 text-gray-800 dark:text-gray-200">Order List</h1>
           
            <ul className={styles.orderList}>
                {orders.map((order, index) => (
                    <li key={index} className={styles.orderItem}>
                        <div className={styles.orderName}>{order.name}</div>
                        <div className={styles.orderDate}>Date: {order.date}</div>
                        <div className={styles.orderAmount}>Amount: {order.amount}</div>
                    </li>
                ))}
            </ul>
        </div>
    );
}
