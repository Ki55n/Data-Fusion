'use client'
import {NextUIProvider} from "@nextui-org/react";

export default function Providers( {children}: any ) {
    return (
        <NextUIProvider>
            {children}
        </NextUIProvider>
    )
}
