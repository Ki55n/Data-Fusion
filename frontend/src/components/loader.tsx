//loader
import React from 'react';
import {ImSpinner9} from 'react-icons/im';

export default function Loader() {
    return <div className={'w-full h-full flex justify-center items-center'}>
        <p className={'text-6xl text-white'}>
            Loading...
        </p>
        <ImSpinner9 className={'animate-spin text-primary text-6xl'}/>
    </div>
}
