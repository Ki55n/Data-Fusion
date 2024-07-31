'use client'
import {useFormState, useFormStatus} from "react-dom";
import React from "react";
import {Button} from "@nextui-org/react";
import {saveProduct} from "@/services/catalog";


const initialState: any = {
    message: "",
};

export const CustomerForm = () => {
    const [state, formAction] = useFormState(saveProduct, initialState);
    const [filesName, setFilesName] = React.useState<any>('');

    return <form className={'flex flex-col gap-4'} action={formAction}>
        <label className="">Title *</label>
        <input
            required
            type="text"
            className="form-input mt-1 block w-full"
            name={'title'}
            placeholder="Product name"
        />
        <label className="">Price</label>
        <input
            required
            type="text"
            className="form-input mt-1 block w-full"
            name={'price'}
            placeholder="Product price*"
        />
        <label className="">Description</label>
        <textarea
            required
            className="form-input mt-1 block w-full"
            name={'description'}
        />
        <SaveButton/>
    </form>
}

const SaveButton = () => {
    const { pending } = useFormStatus();
    return <Button
        type="submit"
        className=" w-full"
        color={'primary'}
        isLoading={pending}
        disabled={pending}
    >
        Upload
    </Button>
}
