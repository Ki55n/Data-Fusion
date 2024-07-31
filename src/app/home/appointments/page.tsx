'use client';
import React, { useState } from 'react';

const AppointmentPage = () => {
  const [darkMode, setDarkMode] = useState(true);

  return (
    <div className="w-full h-screen flex items-center justify-center px-5  ">
        <div>
        <div className="w-full max-w-3xl mx-auto p-6 bg-slate-300 shadow-lg rounded-lg border border-gray-200 mt-12">
  <h2 className="text-3xl font-bold text-gray-800 mb-4">Why Book a Data Analysis Appointment?</h2>
  <p className="text-gray-700 mb-4">
    Data analysis is crucial for making informed business decisions. By booking an appointment with our experts, you gain access to tailored insights and actionable recommendations that can drive your success. Here’s why our data analysis services stand out:
  </p>
  <ul className="list-disc list-inside text-gray-700 mb-6">
    <li className="mb-2">
      <span className="font-semibold">Customized Insights:</span> Get analysis tailored to your specific business needs and objectives.
    </li>
    <li className="mb-2">
      <span className="font-semibold">Data-Driven Decisions:</span> Leverage data to make informed decisions that can enhance performance and growth.
    </li>
    <li className="mb-2">
      <span className="font-semibold">Expert Guidance:</span> Benefit from the expertise of our seasoned analysts who will guide you through complex data.
    </li>
    <li className="mb-2">
      <span className="font-semibold">Time Efficiency:</span> Save time by getting comprehensive analysis quickly, allowing you to focus on what matters most.
    </li>
  </ul>
  <p className="text-gray-700">
    Book your appointment today to unlock the full potential of your data and propel your business forward with actionable insights and strategic recommendations.
  </p>
  
</div>

        </div>
    <div className={`${darkMode ? 'bg-black text-white' : 'bg-white text-black'} min-h-screen py-12`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center mb-8">
          {/* Add a toggle switch or button for dark mode if desired */}
          {/* <button
            onClick={() => setDarkMode(!darkMode)}
            className="px-4 py-2 bg-gray-800 text-white rounded hover:bg-gray-700"
          >
            Toggle Dark Mode
          </button> */}
        </div>
        <h1 className="text-4xl font-bold text-center mb-8 ">Book an Appointment</h1>
        <div className={`${darkMode ? 'bg-gray-800' : 'bg-gray-100'} p-8 mb-3 pt-2 pb-4 rounded-lg shadow-lg max-w-md mx-auto`}>
          <form>
            <div className="mb-4">
              <label className="block text-sm font-bold mb-2" htmlFor="name">Name</label>
              <input className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" type="text" id="name" name="name" />
            </div>
            <div className="mb-4">
              <label className="block text-sm font-bold mb-2" htmlFor="email">Email</label>
              <input className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" type="email" id="email" name="email" />
            </div>
            <div className="mb-4">
              <label className="block text-sm font-bold mb-2" htmlFor="date">Date</label>
              <input className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" type="date" id="date" name="date" />
            </div>
            <div className="mb-4">
              <label className="block text-sm font-bold mb-2" htmlFor="time">Time</label>
              <input className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" type="time" id="time" name="time" />
            </div>
            <div className="mb-4">
              <label className="block text-sm font-bold mb-2" htmlFor="message">Message</label>
              <textarea className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" id="message" name="message"></textarea>
            </div>
           
            <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition duration-300">Book Appointment</button>
          </form>
        </div>
      </div>
    </div>
    </div>
  );
};

export default AppointmentPage;
