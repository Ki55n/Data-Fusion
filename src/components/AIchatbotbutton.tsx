// components/AIChatButton.js
import React from 'react';

const AIChatButton = () => {
  return (
    <div className="flex justify-center items-center min-h-screen bg-black">
      <div className="bg-gray-900 p-4 rounded-full flex items-center shadow-lg">
        <button className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-full relative">
          <span className="absolute inset-0 bg-purple-600 opacity-50 rounded-full filter blur-xl"></span>
          <span className="relative z-10">AI Chat</span>
        </button>
        <div className="ml-4 flex space-x-4">
          <button className="bg-gray-800 p-2 rounded-full hover:bg-gray-700">
            {/* Add icon or text for feature 1 */}
          </button>
          <button className="bg-gray-800 p-2 rounded-full hover:bg-gray-700">
            {/* Add icon or text for feature 2 */}
          </button>
          <button className="bg-gray-800 p-2 rounded-full hover:bg-gray-700">
            {/* Add icon or text for feature 3 */}
          </button>
          <button className="bg-gray-800 p-2 rounded-full hover:bg-gray-700">
            {/* Add icon or text for feature 4 */}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AIChatButton;
