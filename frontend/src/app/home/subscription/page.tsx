"use client"
import React, { useState } from 'react';


const SubscriptionPage = () => {
  const [darkMode, setDarkMode] = useState(true);

//   const toggleDarkMode = () => {
//     setDarkMode(!darkMode);
//   };

  return (
    <div className={`${darkMode ? 'bg-black text-white' : 'bg-white text-black'} min-h-screen py-12`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center mb-8">
          <div className="flex items-center">
           
            
          </div>
          {/* <button 
            onClick={toggleDarkMode} 
            className="bg-gray-800 text-white py-2 px-4 rounded-lg hover:bg-gray-700"
          >
            Toggle {darkMode ? 'Light' : 'Dark'} Mode
          </button> */}
        </div>
        <h1 className="text-4xl font-bold text-center mb-8">Subscribe to Our Data Analysis Platform</h1>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Basic Plan */}
          <div className={`${darkMode ? 'bg-gray-800' : 'bg-gray-100'} p-6 rounded-lg shadow-lg`}>
            <h2 className="text-2xl font-bold mb-4">Basic Plan</h2>
            <p className="mb-6">Get access to basic data analysis tools and features.</p>
            <div className="text-3xl font-bold mb-4">$19<span className="text-lg">/month</span></div>
            <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700">Subscribe</button>
          </div>
          {/* Standard Plan */}
          <div className={`${darkMode ? 'bg-gray-800' : 'bg-gray-100'} p-6 rounded-lg shadow-lg`}>
            <h2 className="text-2xl font-bold mb-4">Standard Plan</h2>
            <p className="mb-6">Includes all features from the Basic Plan plus advanced analytics tools.</p>
            <div className="text-3xl font-bold mb-4">$49<span className="text-lg">/month</span></div>
            <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700">Subscribe</button>
          </div>
          {/* Premium Plan */}
          <div className={`${darkMode ? 'bg-gray-800' : 'bg-gray-100'} p-6 rounded-lg shadow-lg`}>
            <h2 className="text-2xl font-bold mb-4">Premium Plan</h2>
            <p className="mb-6">All features included in the Standard Plan plus premium support and more.</p>
            <div className="text-3xl font-bold mb-4">$99<span className="text-lg">/month</span></div>
            <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700">Subscribe</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SubscriptionPage;
