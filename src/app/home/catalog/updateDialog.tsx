import {ImMagicWand} from "react-icons/im";
import {IoMdArrowBack} from "react-icons/io";
import {FaRegCommentDots} from "react-icons/fa";
import {Divider} from "@nextui-org/divider";
import React from "react";

export default function UpdateDialog({item, product}: any) {
    return (<div className={'relative'}>
            <a href={`/home/catalog?product=${item.id}`}><ImMagicWand/></a>
            {
                product == item.id && <div className={'absolute border-primary border-1 w-[540px] h-25 right-0 p-3  bg-slate-950 z-10'}>
                <p className={'flex gap-2'}><a href={`/home/catalog`}><IoMdArrowBack/></a>You can ask anything like, "add more emojis", or "change the color"</p>
                <div className={'flex m-1 mt-4 gap-3'}>
                  <FaRegCommentDots/>
                  <input placeholder={'Ask the AI anything'} className={'w-full'}/>
                </div>
                <Divider orientation={'horizontal'} />
                <p className={'my-2'}>Suggestions</p>
                <ul className={'flex flex-col gap-2'}>
                  <li>Fix Grammar</li>
                  <li>Improve writing</li>
                  <li>Make it punchier</li>
                  <li>Condense</li>
                  <li>Mix it up</li>
                  <li>Improve structure & spacing</li>
                </ul>
              </div>}
        </div>
    )
}
