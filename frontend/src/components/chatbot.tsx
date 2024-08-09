'use client';
import { Button } from "@nextui-org/button";
import { FaMagic } from "react-icons/fa";
import React, { useState } from "react";
import { Modal, Input, Divider } from "@nextui-org/react";
import {ModalBody, ModalContent, ModalHeader} from "@nextui-org/modal";
import {ProductFom} from "@/app/home/catalog/ProductFom";
import { CiUser } from "react-icons/ci";
import { IoIosVolumeHigh } from "react-icons/io";
import { FaRegCopy } from "react-icons/fa";
import { FaRegCircleStop } from "react-icons/fa6";
import { RiRobot3Fill } from "react-icons/ri";
import { IoCubeOutline } from "react-icons/io5";
import { AIchatbotbutton } from "@/components/AIchatbotbutton";
import { AiOutlineSound } from "react-icons/ai";
import { AiFillSound } from "react-icons/ai";
import { CiDatabase } from "react-icons/ci";

import { IoAnalyticsSharp } from "react-icons/io5";
import Speaker from "@/components/speaker";
import { IoMdCube } from "react-icons/io";
export default function Chatbot() {
    const [userInput, setUserInput] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [isOpen, setIsOpen] = useState(false); // Modal state

    const handleInputChange = (event) => {
        setUserInput(event.target.value);
    };

    const handleSendMessage = () => {
        const response = `Hi there! Thanks for your question about ${userInput}`;
        setChatHistory([...chatHistory, { user: true, message: userInput }, { user: false, message: response }]);
        setUserInput("");
    };

    const handleOpenModal = () => setIsOpen(true);
    const handleCloseModal = () => {
        setIsOpen(false)
    };

    if (!isOpen) {
        return          <>   <div className=" space-x-2  border-slate-800 rounded bg-slate-600 absolute inset-x-0 bottom-0 flex items-center justify-center">
            <button><IoCubeOutline/></button>
            <button >
            <AiOutlineSound />
            </button>
            <Button className="bg-primary"
                startContent={<FaMagic/>}
                onClick={handleOpenModal}
                color={'primary'}>AI Chat</Button>
                <button><IoAnalyticsSharp /></button>
               <button> <CiDatabase /></button>

               
        </div>
        
        </>
        
    }
    return (
        <Modal
            isOpen={true}
            size={'full'}
            onClose={handleCloseModal}

        >
            <ModalContent>
            <ModalHeader className="flex justify-center">
  <h2 className="text-2xl font-bold mb-4 text-center">
    Welcome to Data Fusion chatbot
  </h2>
</ModalHeader>
                <div className={'flex flex-col  h-full '}>


                    { chatHistory.length === 0 ? <div className="text-center p-2 h-5/6">
                        {/* Label */}

                        {/* Paragraph */}
                        <p>
                            This is a financial AI chatbot built with Gemini API. You can start a
                            conversation here or try the following questions:
                            <br />
                            - Most popular item
                            <br />
                            - Try data insights
                        </p>
                    </div>:
                    <div className="chat-history overflow-y-auto h-48 border rounded-lg p-2 h-5/6 p-28 relative">
                        {chatHistory.map((message, index) => (
                            <div key={index}  className={`flex gap-2 `}>
                                <div className={`flex w-full mb-5 gap-2 p-2 bg-primary text-white bg-slate-950 text-white`}>
                                    {message.user ? <CiUser className={'grow-0'}/> : <RiRobot3Fill className={'grow-0'}/>}
                                    <p
                                        className={
                                        'grow w-5/6'
                                        }
                                    >{message.message}</p>
                                    <FaRegCopy className={'grow-0'}/>
                                    <Speaker text={message.message} child={<IoIosVolumeHigh className={'grow-0'}/>}/>
                                </div>
                            </div>

                        ))}
                        <div className={'absolute bottom-0 right-0 left-0 flex justify-center p-3'}>
                            <Button
                                startContent={<FaRegCircleStop/>}
                                className={' w-1/6'}
                            >Stop Generating</Button>
                        </div>


                    </div>}
                    <div className={'flex flex-col gap-3 p-3'}>

                        <Input
                            type="text"
                            value={userInput}
                            onChange={handleInputChange}
                            className="rounded-lg  py-2  focus:outline-none focus:ring-2 focus:ring-primary"
                            placeholder="Type your message..."
                        />
                        <Button onClick={handleSendMessage} startContent={<FaMagic />} color="primary">
                            Send
                        </Button>
                    </div>

                </div>
            </ModalContent>
        </Modal>

    );
}
